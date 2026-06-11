import sys
sys.path.append(r'C:\Users\Administrator\OneDrive\PythonLib')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import os
from sklearn.metrics import mean_squared_error, r2_score
from mdls.EFcode.main_earth import *
from utlts.utils import series_to_superised_out_4d, rmse,dd
from visualization import plot_temp,plot_salt,plot_vec,plot_metrics
from matplotlib.pyplot import *
from tools import RecDataset,GetTestData
# def tic():
#     global _start_time
#     _start_time = time.time()

# def toc():
#     elapsed_time = time.time() - _start_time
#     print(f"Elapsed time: {elapsed_time:.4f} seconds")

def create_prediction_ncfile(output_path, time_data, depth_values, lat_grid, lon_grid):
    """
    创建用于存储预测结果的、符合CF规范的NetCDF文件。
    """
    # 定义时间的单位和日历，这是CF规范的关键
    time_units = f"days since {time_data[0].strftime('%Y-%m-%d %H:%M:%S')}"
    time_calendar = 'standard'
    # 将 datetime 对象转换为相对于 units 的数值
    time_numeric = nc.date2num(time_data, units=time_units, calendar=time_calendar)
    
    with nc.Dataset(output_path, 'w') as nc_out:
        # 创建维度
        nc_out.createDimension('time', len(time_data))
        nc_out.createDimension('depth', len(depth_values))
        nc_out.createDimension('lat', lat_grid.shape[0])
        nc_out.createDimension('lon', lon_grid.shape[1])
        
        # 创建坐标变量
        # 注意：数据类型仍然是 'f8' (double), 这是CF规范推荐的
        time_var = nc_out.createVariable('time', 'f8', ('time',))
        # 写入转换后的数值
        time_var[:] = time_numeric
        # 写入正确的元数据
        time_var.units = time_units
        time_var.calendar = time_calendar
        time_var.long_name = 'time'
        
        # ... (lat, lon 和数据变量的创建保持不变) ...
        nc_out.createVariable('lat', 'f4', ('lat',))
        nc_out.variables['lat'][:] = lat_grid[:, 0]
        nc_out.variables['lat'].units = 'degrees_north'
        nc_out.variables['lat'].long_name = 'latitude'

        nc_out.createVariable('lon', 'f4', ('lon',))
        nc_out.variables['lon'][:] = lon_grid[0, :]
        nc_out.variables['lon'].units = 'degrees_east'
        nc_out.variables['lon'].long_name = 'longitude'
        
        # 创建坐标变量
        nc_out.createVariable('depth', 'f4', ('depth',))
        nc_out.variables['depth'][:] = depth_values
        nc_out.variables['depth'].units = 'meters'
        nc_out.variables['depth'].long_name = 'depth below sea level'
        
        # 创建预测变量 - 温度
        nc_out.createVariable('predicted_t', 'f4', 
                            ('time', 'depth', 'lat', 'lon'), 
                            fill_value=np.nan, zlib=True)
        nc_out.variables['predicted_t'].long_name = 'Predicted sea water potential temperature'
        nc_out.variables['predicted_t'].units = 'degrees Celsius'
        nc_out.variables['predicted_t'].standard_name = 'sea_water_potential_temperature'
        
        # 创建预测变量 - 盐度
        nc_out.createVariable('predicted_s', 'f4', 
                            ('time', 'depth', 'lat', 'lon'), 
                            fill_value=np.nan, zlib=True)
        nc_out.variables['predicted_s'].long_name = 'Predicted sea water practical salinity'
        nc_out.variables['predicted_s'].units = '1e-3'  # 实用盐度单位
        nc_out.variables['predicted_s'].standard_name = 'sea_water_practical_salinity'
        
        # 创建观测变量 - 温度
        nc_out.createVariable('observed_t', 'f4', 
                            ('time', 'depth', 'lat', 'lon'), 
                            fill_value=np.nan, zlib=True)
        nc_out.variables['observed_t'].long_name = 'Observed sea water potential temperature'
        nc_out.variables['observed_t'].units = 'degrees Celsius'
        nc_out.variables['observed_t'].standard_name = 'sea_water_potential_temperature'
        
        # 创建观测变量 - 盐度
        nc_out.createVariable('observed_s', 'f4', 
                            ('time', 'depth', 'lat', 'lon'), 
                            fill_value=np.nan, zlib=True)
        nc_out.variables['observed_s'].long_name = 'Observed sea water practical salinity'
        nc_out.variables['observed_s'].units = '1e-3'  # 实用盐度单位
        nc_out.variables['observed_s'].standard_name = 'sea_water_practical_salinity'

# 固定随机种子
def set_seed(seed=324):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

class DynamicMSECRPSLoss(nn.Module):
    def __init__(self, warmup_steps=50, reduction='mean', eps=1e-8):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.reduction = reduction
        self.eps = eps
        self.step = 0

    def forward(self, target, prediction, mask=None, update_step=True):
        mask_bool = None
        if mask is not None:
            mask_bool = mask.bool()
            if mask_bool.any():
                target_valid = torch.masked_select(target, mask_bool)
                pred_valid = torch.masked_select(prediction, mask_bool)
                mse_loss = F.mse_loss(pred_valid, target_valid, reduction=self.reduction)
            else:
                mse_loss = prediction.new_tensor(0.0)
            mask_float = mask_bool.to(prediction.dtype)
            target_for_crps = target * mask_float
            pred_for_crps = prediction * mask_float
        else:
            mse_loss = F.mse_loss(prediction, target, reduction=self.reduction)
            target_for_crps = target
            pred_for_crps = prediction
        
        if self.step < self.warmup_steps:
            loss = mse_loss
        else:
            crps_val = crps_loss(pred_for_crps.squeeze(0), target_for_crps.squeeze(0))
            weight_sum = mse_loss + crps_val + prediction.new_tensor(self.eps)
            w_mse = crps_val / weight_sum
            w_crps = mse_loss / weight_sum
            loss = w_mse * mse_loss + w_crps * crps_val

        if update_step:
            self.step += 1

        return loss

class Model():
    def __init__(self, params, loading_path=None, set_device=4):
        if params['network_name'] == 'sa_convlstm':
            self.model = Encode2Decode_SAconvLSTM_MultiStep(params).to(params['device'])
        elif params['network_name'] == 'convlstm':
            self.model = Encode2Decode_convLSTM_MultiStep(params).to(params['device'])
        elif params['network_name'] == 'spatial_attention':
            self.model = Encode2Decode_SpatialAttention(params).to(params['device'])
        elif params['network_name'] == 'EF':
            self.model = CuboidWaveModel(total_num_steps = 20,oc_file = params['config_file'], save_dir= './').torch_nn_module
        # 添加对TransformerUNet的支持
        elif params['network_name'] == 'TUNet':
            # 确定输入通道数，根据您的数据特征
            input_channels = len(params['vlist'])
            # 定义通道配置，您可以根据需要调整这些值
            channels = [input_channels, 64, 128, 256, 512]  # 示例配置
            #最后一个channel为输出维度！
            self.model = TransformerUNet(channels,out_channels=19).to(params['device'])        
        
        #self.model = BalancedDataParallel(1, self.model, dim=0).cuda()
        self.model.to(device)
        #self.model = torch.compile(self.model)

        self.loss = params['loss']
        if self.loss == 'SSIM':
            self.criterion = SSIM().to(device)
        elif self.loss == 'L2':
            self.criterion = nn.HuberLoss()
        elif self.loss == 'MSE+CRPS':
            self.criterion = DynamicMSECRPSLoss()
        else:
            self.criterion = nn.L1Loss()
        #self.output = params['output_dim']
        self.device = params['device']
        self.optim = optim.Adam(self.model.parameters(), lr=params['lr'])
        self.params = params


# 配置
set_seed()
BATCH_SIZE = 24
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_name = 'EF'
fname = r"D:\Github\DORS\Data\GLORYS\cmems_glorys_2010-2024.nc" #读取深度
vname_postfix = '_glor'
data_folder = r'D:\Github\DORS\Data\GLORYS' #ncfile folder

# 添加命令行参数解析
import argparse  # 添加argparse模块
parser = argparse.ArgumentParser(description='Run prediction with specified model prefix')
parser.add_argument('--nn_prefix', type=str, default='GLOM_EF_temp3d13gradf_ADTOIAll_mask_Xnorm_gradY_MSE+CRPS',
                   help='Neural network file name prefix')
nn_prefix = parser.parse_args().nn_prefix   #文件名前缀
#nn_prefix = 'GLO_EF_T5gradf_ADTOIAll_mask_Xnorm'   #文件名前缀
# 1. 加载模型
model_name = "".join(nn_prefix)
target = 'temp3d' if 'temp3d' in model_name else 'salt3d'
model_name = model_name.replace('temp3d','')  #整合TS重建
model_name = model_name.replace('salt3d','')  #
#nn_name = f"C:/Github/DORS/res_GLOM/{model_name}/{nn_prefix}_epoch199.h5" #神经网络文件
nn_name = f"D:/Github/DORS/res_GLO/{model_name}/{target}_epoch199.h5" #神经网络文件
output_name = f'res_GLO/{model_name}_epoch199predictions_ts.nc'

nc_file = nc.Dataset(fname,'r')
index = np.array([4,7,11,14,16,18,21,24,25,26,27,30,32,34,37,39,42,44,46]) #5-1000m
depth = nc_file['depth'][index]
lon=nc_file['longitude'][:]
lat= nc_file['latitude'][-121:] if nn_prefix.startswith('GLOM') else nc_file['latitude'][:]
(lon,lat)=np.meshgrid(lon,lat)
#
import xarray as xr
ds = xr.open_dataset(fname)  # xarray自动处理cftime时间
time = ds['time'][-1096:]
import pandas as pd
time = pd.to_datetime(time).to_pydatetime()
# 生成nc文件
if not os.path.exists(output_name):
    create_prediction_ncfile(output_name, time_data=time, depth_values=depth,
                    lat_grid=lat,lon_grid=lon)
    print(f'Create {output_name} nc file!')
# 初始化结果数组
N = len(depth)  # 深度层数量
# 初始化 results 字典
NN = torch.load(nn_name, map_location=device, weights_only=False)
model = NN.model
model.eval()
params = NN.params
if 'input_norm' not in params.keys():
    params['input_norm']=False
    params['output_norm']=False

# 2. 加载数据集
dataset = RecDataset(data_folder, '', vname_postfix, index, params,BATCHING_DATASET=8)

# 测试集划分（与训练一致后20%）
total_len = len(dataset)

#test_indices = np.arange(int(total_len * 0.8), total_len)
#test_loader = DataLoader(dataset, batch_size=1, sampler=test_indices)
test_sampler  = np.arange(547, total_len) 
test_loader   = DataLoader(dataset, batch_size=1, sampler=test_sampler)

# 3. 推理与指标
y_true_list = []
y_pred_list = []

with torch.no_grad():
    # 初始化列表，存储所有批次的真实值和预测值
    y_true_all = []  # 最终维度 [总时间步数, 1, H, W, 2]
    y_pred_all = []  # 最终维度 [总时间步数, 1, H, W, 2]

    #for idx in test_sampler:
    for idx, (batch_in,batch_out) in enumerate(test_loader):
        #batch_in, batch_out = dataset.__getitem__(idx)
        if nn_prefix.startswith('GLOM'):
            batch_in = batch_in[:,:,-121:]
            batch_out = batch_out[:,:,-121:]
            
        print(idx)
        batch_in = batch_in.to(dtype=torch.float, device=device).squeeze(0)    
        if batch_out is not None:         
            batch_out = batch_out.to(dtype=torch.float, device=device).squeeze(0)             
        
        if params['output_norm'] and params['target']=='temp3d':
            mean = params['thetao_global_mean'][:]
            std  = params['thetao_global_std'][:]
        elif params['output_norm'] and params['target']=='salt3d':
            mean = params['so_global_mean'][:]
            std  = params['so_global_std'][:]
        else:
            mean = np.array([0]);std=np.array([1])
        
        pred = model(batch_in) * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
        batch_out = batch_out * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)

        # 将当前批次的数据添加到列表中
        y_true_all.append(batch_out.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

    # 合并所有批次的数据（沿时间维拼接）
    y_true_all = np.concatenate(y_true_all, axis=0)  # 维度 [总时间步数, 1, H, W, 2]
    y_pred_all = np.concatenate(y_pred_all, axis=0)  # 维度 [总时间步数, 1, H, W, 2]
        
    y_pred_all[y_true_all==0]=np.nan
    y_true_all[y_true_all==0]=np.nan
    y_pred_all = np.swapaxes(y_pred_all,4,1).squeeze()
    y_true_all = np.swapaxes(y_true_all,4,1).squeeze()
    # 将结果写入NetCDF文件
    inx = 10
    #tinx = time.get_index('time').get_loc('2023-11-22')
    tinx = pd.DatetimeIndex(time).get_loc('2023-11-22')
    with nc.Dataset(output_name, 'a') as nc_out:
        # 写入所有深度的数据
        if NN.params['target'] == 'temp3d':
            nc_out.variables['predicted_t'][:] = y_pred_all
            nc_out.variables['observed_t'][:] = y_true_all
            plot_temp(lon,lat, y_true_all[tinx,inx], y_pred_all[tinx,inx], time = time[tinx], z = depth[inx],des=nn_prefix)
        elif NN.params['target'] == 'salt3d':
            nc_out.variables['predicted_s'][:] = y_pred_all
            nc_out.variables['observed_s'][:] = y_true_all
            plot_salt(lon,lat, y_true_all[tinx,inx], y_pred_all[tinx,inx], time = time[tinx], z = depth[inx],des=nn_prefix)
print(f'Save to {output_name}')    
overall_RMSE = np.nanmean((y_pred_all-y_true_all)**2)**0.5
print(f'Overall RMSE={overall_RMSE:.4f}')
z_RMSE = np.nanmean((y_pred_all-y_true_all)**2, axis=(0,2,3))**0.5
semilogy(z_RMSE,depth)
title(f'{output_name}')
os.makedirs(f'{nn_prefix}',exist_ok=True)
savefig(f'res_GLO\{model_name}\{params['target']}_RMSE_profile.png')

