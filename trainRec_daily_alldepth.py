import sys
sys.path.append(r'H:\TS reconstruction')# by LWF, user-defined models
import numpy as np
from matplotlib import *
from datetime import datetime,timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import netCDF4 as nc
import xarray as xr
import sys
from torch.autograd import Variable
from matplotlib.pyplot import *
import scipy.io as sio 
from torch.utils.data import TensorDataset,DataLoader, random_split
# import h5py
#from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
from torch import optim
from tqdm import tqdm
#from params import *
from mdls.EFcode.main_earth import *
from mdls.TransformerUnet.TransformerUNet import *
from utlts.utils import series_to_superised_out_4d,dd,rmse
from utlts.time_manager import Profiler
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#from torchinfo import summary
from utlts.data_parallel_my_v2 import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
from tools import RecDataset,crps_loss
import torch.fft

import os
import shutil
from datetime import datetime

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

def adjust_learning_rate(lr, optimizer, epoch, lr_changestep):
    """Sets the learning rate to the initial LR decayed by 10 every args.step epochs"""
    if epoch > 1:
        lr = lr * (0.9 ** (epoch // lr_changestep))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

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
            self.criterion = nn.MSELoss()
        elif self.loss == 'MSE+CRPS':
            self.criterion = DynamicMSECRPSLoss()
        else:
            self.criterion = nn.L1Loss()
        #self.output = params['output_dim']
        self.device = params['device']
        self.optim = optim.Adam(self.model.parameters(), lr=params['lr'])
        self.params = params
    
##start here

set_seed()
epochs = 1
BATCH_SIZE = 10
loss_accu_step = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 添加命令行参数解析
import argparse  # 添加这行
import yaml      # 添加这行
parser = argparse.ArgumentParser(description='Training script with YAML configuration')
parser.add_argument('--target', type=str, help='temp3d or salt3d',default='temp3d')
parser.add_argument('--params', type=str, help='YAML configuration file path',default='experiments/Case11M.yaml')
args = parser.parse_args()
params_file = args.params # 解析YAML文件,一次性导入所有参数
with open(params_file, 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
params['target']=args.target

# 补齐旧 Case 配置中可能没有的字段，避免 KeyError
params.setdefault('spec', False)
params.setdefault('gradY', False)
params.setdefault('gradZ', False)
params.setdefault('decoZ', False)
params.setdefault('continue_training', False)
params.setdefault('batch_size', 10)

print(f"params = {params}")
#params['output_norm']=True
# 从params获取数据类型
datatype = params.get('datatype', torch.float32)
print(f"Using datatype: {datatype}")

fname = r"D:/subsurface marine heatwaves data/GLORYS/cmems_glorys_2010-2023.nc" #读取深度
#index = np.array([0,2,4,7,11,14,16,18,21,24,25,27,30,32,34,37,39,42,44,46,50,53])
index = np.array([4,7,11,14,16,18,21,24,25,26,27,30,32,34,37,39,42,44,46]) #5-1000m
depth = nc.Dataset(fname)['depth'][index]

vname_postfix = '_glor'
data_folder = r'D:/subsurface marine heatwaves data/GLORYS' #ncfile folder

model_name   = f"res_GLO/GLO_{params['network_name']}_{len(params['vlist'])}gradf_ADTOIAll"   #文件名前缀

#print('SCS!')
loss_weight = 1
if params['mask']:
    model_name+='_mask'
if params['output_norm']:
    model_name+='_Ynorm'
if params['input_norm']:
    model_name+='_Xnorm'
if params.get('spec', False):
    model_name+='_spec'
    loss_weight+=1
if params.get('gradY', False):
    model_name+='_gradY'
    loss_weight+=1
if params.get('gradZ', False):
    model_name+='_gradZ'
    loss_weight+=1
if params.get('loss', '') == 'MSE+CRPS':
    model_name+='_MSE+CRPS'
# if params['decoZ']:
#     model_name+='_decoZ'

# ==================== 实验记录和路径设置 ====================
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
script_name = os.path.basename(current_script_path)

# 创建experiments文件夹（如果不存在）
experiment_dir = os.path.join(os.path.dirname(current_script_path), model_name)
os.makedirs(experiment_dir, exist_ok=True)
# 将输出都保存到这里
savePath = experiment_dir

# 复制当前脚本到实验文件夹
shutil.copy2(current_script_path, os.path.join(experiment_dir, script_name))
# 复制参数文件到实验文件夹
shutil.copy(params_file, savePath)
print(f"当前实验文件夹: {experiment_dir}")
#
# data_folder = r'D:\Github\DORS\ORAS'
# vname_postfix = '_oras'

# 计算平均TS的代码段
# ds = xr.open_dataset(r'D:/Github/DORS/Data/GLORYS/bak/GLORYS_2020.nc')
# thetao_global_mean = ds['thetao_glor'].mean(dim=['time','latitude', 'longitude']).values
# so_global_mean = ds['so_glor'].mean(dim=['time', 'latitude', 'longitude']).values
# thetao_global_std = ds['thetao_glor'].std(dim=['time','latitude', 'longitude']).values
# so_global_std = ds['so_glor'].std(dim=['time', 'latitude', 'longitude']).values

# dd=nc.Dataset(fname)['depth'][:]
# new_ds = xr.Dataset({
#     'thetao_global_mean': thetao_global_mean,
#     'so_global_mean': so_global_mean,
#     'thetao_global_std':thetao_global_std,
#     'so_global_std':so_global_std,
#     'depth': dd})
# new_ds.to_netcdf('D:/subsurface marine heatwaves data/GLORYS\bak\global_meanstd_2020.nc')

ds_mean = xr.open_dataset(r'D:/subsurface marine heatwaves data/GLORYS/bak/global_meanstd_2020.nc', engine='netcdf4')
params['thetao_global_mean']=ds_mean['thetao_global_mean'][index].values
params['so_global_mean']=ds_mean['so_global_mean'][index].values
params['thetao_global_std']=ds_mean['thetao_global_std'][index].values
params['so_global_std']=ds_mean['so_global_std'][index].values

inx = -1 #all depth
# for inx in range(0,len(depth)):
#     Level = index[inx] #true Level
fname_year = ''
dataset = RecDataset(data_folder, fname_year, vname_postfix, index, params, BATCHING_DATASET=10)
example, y1 = dataset.__getitem__(36)
figure(figsize=(8,6))
pcolor(dd(example[0,0,:,:,-4]));title(params['vlist'][-4]);colorbar();show()
pcolor(dd(example[0,0,:,:,6]));title(params['vlist'][6]);colorbar();show()

if params['continue_training']:
    pre_model = r"D:\GitHub\DORS\res_GLOM\GLOM_EF_13gradf_ADTOIAll_mask_Xnorm_gradY\GLOM_EF_temp3d13gradf_ADTOIAll_mask_Xnorm_gradY_epoch49.h5"
    NN = torch.load(pre_model, map_location=device, weights_only=False)
    print(f'Use pretrained model {pre_model}')
    NN.criterion = DynamicMSECRPSLoss()
    avg_train_losses = NN.params['train_loss']
    avg_valid_losses = NN.params['valid_loss']
    NN.params = params
    start_epoch = 50
    val_epoch_loss=np.sqrt(avg_valid_losses[-1])
else:
    NN = Model(params)  
    print(f'Use new model')
    avg_train_losses = []
    avg_valid_losses = []
    start_epoch = 0
    val_epoch_loss=0

#制作数据集
# trainset, valset = random_split(
#         dataset, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)  # 设置随机种子保证可重复性
#     )
split = np.int64(len(dataset)*0.8)
train_index = np.arange(0, split)
val_index   = np.arange(split, len(dataset)-1)
#如果是RecDatasetBatch，以下batch_size=1
train_loader = DataLoader(dataset, batch_size=1, pin_memory=False, num_workers=0, 
                          sampler=torch.utils.data.SubsetRandomSampler(train_index)) 
val_loader   = DataLoader(dataset, batch_size=1, pin_memory=False, num_workers=0,
                          sampler=val_index)

lrOut = params['lr']

# kx = np.fft.fftshift(np.fft.fftfreq(161)) * 161  # x方向波数
# ky = np.fft.fftshift(np.fft.fftfreq(221)) * 221  # y方向波数
# kx,ky = np.meshgrid(kx,ky,indexing='ij')

scaler = torch.amp.GradScaler('cuda') #half 1


with tqdm(total=epochs-start_epoch, desc=f'{model_name}') as pbar:
   
    for epoch in range(start_epoch ,epochs):        
        epoch_loss = 0
        
        train_losses, val_losses = [], []
        NN.model.train()

        #
        for idx, (batch_in,batch_out) in enumerate(train_loader):
            #with profiler.profile("1. Data Loading"):    
            if model_name.startswith('GLOM'):
                batch_in = batch_in[:,:,-121:]
                batch_out = batch_out[:,:,-121:]
            NN.optim.zero_grad()
            batch_in = batch_in.to(dtype=datatype, device=device).squeeze(0) #如果是RecDatasetBatch，这里要squeeze(0);否则squeeze(1)
            #
            #mask[:,:,:,100:,75:]= False
            batch_out = batch_out.to(dtype=datatype, device=device).squeeze(0)
            mask=(batch_out)!=0 #mask要在标准化前
            with torch.amp.autocast('cuda',dtype=datatype):
                
                data_out = NN.model(batch_in) #BTXYC
                
                if params['mask']:
                    mask = mask.to(device).squeeze(0)
                    if params.get('loss', '') == 'MSE+CRPS':
                        
                        loss = NN.criterion(batch_out,data_out,mask)
                    else:
                        tmp1 = torch.masked_select(batch_out,mask)
                        tmp2 = torch.masked_select(data_out,mask)
                        loss = NN.criterion(tmp1,tmp2)
                    
                else:
                    loss = NN.criterion(batch_out,data_out)
                
                #print(f'Loss before={loss.item()}')
                loss1=0
                if params.get('spec', False) and epoch>10:
                    from tools_filter import analyze_velocity_field_spectrum_latlon 
                    kx, ky, est_spec = analyze_velocity_field_spectrum_latlon(torch.permute(data_out,(0,1,4,2,3))) 
                    _,_,  true_spec =  analyze_velocity_field_spectrum_latlon(torch.permute(batch_out,(0,1,4,2,3))) 
                    est_spec = torch.log(est_spec + 1e-8 )
                    true_spec = torch.log(true_spec + 1e-8 )
                    tmp    = NN.criterion(true_spec,est_spec, update_step=False) 
                    loss1 += tmp * loss.item()/(tmp.item()+1e-10)
                    kx = dd(kx);ky = dd(ky)
                    if idx==0:
                        figure(figsize=(10,8)) #*3.9313812+19.58339
                        subplot(221);pcolor(dd(batch_out[0,0,:,:,10]),vmin=10,vmax=30);colorbar()
                        subplot(222);pcolor(ky,kx,dd(true_spec),vmin=0,vmax=20);colorbar()
                        subplot(223);pcolor(dd(data_out[0,0,:,:,10]),vmin=10,vmax=30);colorbar()
                        subplot(224);pcolor(ky,kx,dd(est_spec),vmin=0,vmax=20);colorbar()
                        suptitle(f'{epoch} spec')
                        show()
                
                if params.get('gradY', False) and epoch>10:
                    true_dtdx = torch.log(torch.gradient(batch_out,dim=2)[0]**2 + 1e-8)
                    true_dtdy = torch.log(torch.gradient(batch_out,dim=3)[0]**2 + 1e-8)
                    est_dtdx = torch.log(torch.gradient(data_out,dim=2)[0]**2 + 1e-8)
                    est_dtdy = torch.log(torch.gradient(data_out,dim=3)[0]**2 + 1e-8)
                    
                    mask_grad = torch.isfinite(true_dtdx) & torch.isfinite(true_dtdy) & (true_dtdy!=0)
                    #loss += NN.criterion(true_grad[mask_grad],est_grad[mask_grad])
                    if params.get('loss', '') == 'MSE+CRPS':
                        tmp    =  NN.criterion(true_dtdx,est_dtdx,mask_grad, update_step=False) \
                                 +NN.criterion(true_dtdy,est_dtdy,mask_grad, update_step=False)   
                    else:
                        tmp    =  NN.criterion(true_dtdx[mask_grad],est_dtdx[mask_grad], update_step=False) \
                                 +NN.criterion(true_dtdy[mask_grad],est_dtdy[mask_grad], update_step=False)   
                    loss1 += tmp * loss.item()/(tmp.item()+1e-10)
                    
                    if idx==0:
                        figure(figsize=(10,8)) #*3.9313812+19.58339
                        subplot(221);pcolor(dd(batch_out[0,0,:,:,3]),vmin=10,vmax=35);colorbar()
                        subplot(222);pcolor(dd(true_dtdx[0,0,:,:,3]),vmin=-6,vmax=2);colorbar()
                        subplot(223);pcolor(dd(data_out[0,0,:,:,3]),vmin=10,vmax=35);colorbar()
                        subplot(224);pcolor(dd(est_dtdx[0,0,:,:,3]),vmin=-6,vmax=2);colorbar()
                        suptitle(f'{epoch} gradY')
                        show()
                
                if params.get('gradZ', False) and epoch>10:
                    dz = torch.gradient(torch.from_numpy(depth))[0].to(device)
                    true_dydz = torch.gradient(batch_out,dim=-1)[0]/dz
                    est_dydz  = torch.gradient(data_out,dim=-1)[0]/dz 
                    mask_grad = torch.isfinite(true_dydz) & (true_dydz!=0)
                    if params.get('loss', '') == 'MSE+CRPS':
                        tmp    = NN.criterion(true_dydz,est_dydz, mask_grad, update_step=False)
                    else:
                        tmp    = NN.criterion(true_dydz[mask_grad],est_dydz[mask_grad], update_step=False)
                        
                    loss1 += tmp * loss.item()/(tmp.item()+1e-10)
                    if idx==0:
                        figure(figsize=(10,8)) #*3.9313812+19.58339
                        subplot(221);plot(dd(batch_out[0,0,60,160,:]),-depth);plot(dd(data_out[0,0,60,160,:]),-depth)
                        subplot(222);plot(dd(true_dydz[0,0,60,160,:]),-depth);plot(dd(est_dydz[0,0,60,160,:]),-depth)
                        subplot(223);plot(dd(batch_out[0,0,80,160,:]),-depth);plot(dd(data_out[0,0,80,160,:]),-depth)
                        subplot(224);plot(dd(true_dydz[0,0,80,160,:]),-depth);plot(dd(est_dydz[0,0,80,160,:]),-depth)
                        suptitle(f'{epoch} gradZ')
                        show()
                #if params['grad_z']:
                # # not scale
                # loss.backward()
                # NN.optim.step()  # Does the update
                # NN.optim.zero_grad()  # zero the gradient buffers, was NN.optim.zero
                #
                # scale
                
                if epoch>10:
                    loss = (loss + loss1)/loss_weight
                else:
                    loss = loss
                
                scaler.scale(loss).backward() # backward using scaler
                scaler.step(NN.optim) # Does the update using scaler
                scaler.update()
                NN.optim.zero_grad()  # 更新后清零梯度
            
            train_losses.append(loss.item())
            epoch_loss += loss.item()/len(train_loader)
            #tic = time.perf_counter()   
        
        #after all data updated, update pbar (counting for epoch loop)
        avg_train_losses.append(np.average(train_losses))
        NN.params['train_loss'] = avg_train_losses
        lrOut = adjust_learning_rate(NN.params['lr'], NN.optim, epoch-start_epoch, NN.params['lrStep'])
        pbar.set_postfix(**{'Loss(validation)': np.sqrt(val_epoch_loss), 'Loss(Total)': np.sqrt(epoch_loss), 'lr': lrOut})
        pbar.update()

        
        if np.mod(epoch + 1, NN.params['saveStep']) == 0 or epoch==epochs-1:
            val_epoch_loss = 0
            NN.model.eval()
            with torch.no_grad():
                      
                #for idx in val_index:
                for idx, (batch_in,batch_out) in enumerate(val_loader):
                    #print(idx)
                    #(batch_in,batch_out) = dataset.__getitem__(idx)
                    if model_name.startswith('GLOM'):
                        batch_in = batch_in[:,:,-121:]
                        batch_out = batch_out[:,:,-121:]
                    batch_in = batch_in.to(dtype=datatype, device=device).squeeze(0)             
                    
                    #先mask，再标准化
                    mask=(batch_out)!=0
                    mask=mask.to(device)
                    #mask[:,:,:,100:,75:]= False
                    #
                    #batch_out = (batch_out - mean)/std 
                    batch_out = batch_out.to(dtype=datatype, device=device).squeeze(0) 
                    data_out = NN.model(batch_in)  #* torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                    
                    #mask=(batch_out+data_out)!=0
                    if params['mask']:
                        
                        if params.get('loss', '') == 'MSE+CRPS':
                            
                            loss = NN.criterion(batch_out,data_out,mask, update_step=False)
                            
                        else:
                            tmp1 = torch.masked_select(batch_out,mask)
                            tmp2 = torch.masked_select(data_out,mask)
                            loss = NN.criterion(tmp1,tmp2)
                    
                    else:
                        loss = NN.criterion(batch_out,data_out, update_step=False)
                        
                    val_epoch_loss += loss.item()/len(val_loader)
            
            avg_valid_losses.append(val_epoch_loss)
            NN.params['valid_loss'] = avg_valid_losses
            
            figure(1);
            semilogy(np.arange(epoch+1),avg_train_losses);
            semilogy(np.arange(NN.params['saveStep']-1, epoch+2, NN.params['saveStep']), avg_valid_losses);
            title(model_name)
            show()
            # 
            #figure(2);plot(dd(batch_out.flatten()),dd(data_out.flatten()),'b.');show()
            
            print(f'CRPS =  {dd(crps_loss(data_out,batch_out))} ')
            
            outname = model_name + f"/{params['target']}_epoch{epoch}.h5"     #神经网络文件
            torch.save(NN, outname)

print('Finish and save to '+outname)
