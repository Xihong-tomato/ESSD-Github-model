# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:31:43 2025

@author: Scenty
"""
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
import sys
from torch.autograd import Variable
from matplotlib.pyplot import *
import scipy.io as sio 
from torch.utils.data import TensorDataset,DataLoader, random_split
# import h5py
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
from torch import optim
from tqdm import tqdm
#from params import *
from mdls.EFcode.main_earth import *
from utlts.utils import series_to_superised_out_4d,dd,rmse
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#from torchinfo import summary
from utlts.data_parallel_my_v2 import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

import os
import shutil
from datetime import datetime
import xarray as xr

s=torch.cuda.Stream()
#RecDataset不需要输入batch，每次读取一个batch。但是由于nc文件的限制（无法多个进程同时访问），读取效率很低。
#RecDatasetBatch则需要输入batch，每次读取batch个数据。
class RecDataset(Dataset):  
    def __init__(self, data_folder, fname_year, vname_postfix, Level, params, BATCHING_DATASET=None):  
        self.data_folder = data_folder  
        self.data_file_list = [file for file in sorted(os.listdir(data_folder)) if file.endswith(fname_year+'.nc')]  
        self.dataset_dict = self.create_dataset_dict()
        self.Level = Level
        self.BATCH_SIZE = BATCHING_DATASET if BATCHING_DATASET is not None else None
        self.vname_postfix=vname_postfix
        self.params = params
        
        filename = self.dataset_dict[0] #only one file
        file_path = os.path.join(self.data_folder, filename)  
        with nc.Dataset(file_path, 'r') as nc_file:  
            L=len(nc_file.variables['time'][:])
        
        self.length = L

    def create_dataset_dict(self):  
        dataset_dict = {}  
        idx = 0
        #start_file_index = Lag_month + BATCH_SIZE
        for current_Y_index in range(len(self.data_file_list)): #start from zero now，只有一个nc文件才起作用
            filename = os.path.join(self.data_folder, self.data_file_list[current_Y_index])
            dataset_dict[idx] = filename  
            idx += 1  
        return dataset_dict  

    def __len__(self):  
        filename = self.dataset_dict[0] #only one file
        file_path = os.path.join(self.data_folder, filename)  
        with nc.Dataset(file_path, 'r') as nc_file:  
            L=len(nc_file.variables['time'][:])
        
        if self.BATCH_SIZE is not None:
            L=L//self.BATCH_SIZE
            
        return L

    def __getitem__(self, idx):
        params = self.params
        filename = self.dataset_dict[0]
        file_path = os.path.join(self.data_folder, filename)
        if self.BATCH_SIZE is not None:
            start = idx * self.BATCH_SIZE
            end = min( (idx + 1) * self.BATCH_SIZE, self.length)
        else:
            start = idx
            end = idx+1
        Level = self.Level
        vname_postfix = self.vname_postfix
        #print(f'idx {idx}:{start}-{end}')
        
        X_npy = ''
        for var in params['vlist']:
            X_npy += '+'+ var
        X_npy = 'X_' + X_npy[1:] + '_idx(' + str(start) + '-' + str(end) + ')'
        if self.params['input_norm']:
            X_npy = X_npy + '_norm.npy'
        else:
            X_npy = X_npy + '_nonorm.npy'

        Y_npy = 'Y_' + params['target'] + '_idx(' + str(start) + '-' + str(end) + ')'
        if self.params['output_norm']:
            Y_npy = Y_npy + '_norm.npy'
        else:
            Y_npy = Y_npy + '_nonorm.npy'

        savepath = r'./Data/temp'
        if os.path.exists(os.path.join(savepath, X_npy)):
            #print('loading local X.npy')
            X = np.load(os.path.join(savepath, X_npy)).astype(np.float32)
        else:
            # 1. 初始化特征列表
            X_list = []
            
            with nc.Dataset(file_path, 'r') as nc_file:
                # 按vlist顺序读取并处理每个变量
                for var in params['vlist']:
                    if var == 'sst':
                        data = nc_file.variables['thetao'+vname_postfix][start:end, 0].data[np.newaxis]  # [1, B, X, Y]
                        if params['input_norm']:
                            raise NotImplementedError("Input normalization not implemented")
                        X_list.append(data)
                    
                    elif var == 'sss':
                        data = nc_file.variables['so'+vname_postfix][start:end, 0].data[np.newaxis]  # [1, B, X, Y]
                        if params['input_norm']:
                            raise NotImplementedError("Input normalization not implemented")
                        X_list.append(data)
                        
                    elif var == 'zos':
                        data = nc_file.variables['zos'+vname_postfix][start:end].data[np.newaxis]  # [1, B, X, Y]
                        if params['input_norm']:
                            raise NotImplementedError("Input normalization not implemented")
                        X_list.append(data)
                        
                    elif var == 'mask':
                        data = nc_file.variables['thetao'+vname_postfix][start:end, Level].mask[np.newaxis]  # [1, B, X, Y]
                        if params['input_norm']:
                            raise NotImplementedError("Input normalization not implemented")
                        X_list.append(data)
                    
                    elif var == 'lon':
                        lon = nc_file.variables['longitude'][:]
                        lat = nc_file.variables['latitude'][:]
                        lon_grid, _ = np.meshgrid(lon, lat)
                        data = np.broadcast_to(lon_grid[np.newaxis, np.newaxis], 
                                             (1, end-start, *lon_grid.shape))  # [1, B, X, Y]
                        if params['input_norm']:
                            mean, std = 132.5, 15.9491379077
                            data = (data - mean) / std
                        X_list.append(data)
                    
                    elif var == 'lat':
                        lon = nc_file.variables['longitude'][:]
                        lat = nc_file.variables['latitude'][:]
                        _, lat_grid = np.meshgrid(lon, lat)
                        data = np.broadcast_to(lat_grid[np.newaxis, np.newaxis], 
                                             (1, end-start, *lat_grid.shape))  # [1, B, X, Y]
                        if params['input_norm']:
                            mean, std = 20, 11.6189500386
                            data = (data - mean) / std
                        X_list.append(data)
                                
            # 2. 读取外部数据
            if 'time_sin' in params['vlist']:
                with xr.open_dataset(file_path) as ds:
                    time = np.sin(ds.time[start:end].dt.dayofyear.values /365 * 2 * np.pi)
                    data = np.broadcast_to(time[np.newaxis, :, np.newaxis, np.newaxis], 
                                               data.shape)  # [1, B, X, Y]
                    X_list.insert(params['vlist'].index('time_sin'), data)  # 插入到原vlist指定位置
                    
            if 'time_cos' in params['vlist']:
                with xr.open_dataset(file_path) as ds:
                    time = np.cos(ds.time[start:end].dt.dayofyear.values /365 * 2 * np.pi)
                    data = np.broadcast_to(time[np.newaxis, :, np.newaxis, np.newaxis], 
                                               data.shape)  # [1, B, X, Y]
                    X_list.insert(params['vlist'].index('time_cos'), data)  # 插入到原vlist指定位置
                
            if 'adt' in params['vlist']:
                with nc.Dataset(r"D:/subsurface marine heatwaves data/ADT/calibrated_adt_from125_2010-2023.nc", 'r') as nc_file:
                    data = nc_file.variables['calibrated_adt'][start:end].data[np.newaxis]  # [1, B, X, Y]
                    if params['input_norm']:
                        mean, std = 0.740288219647, 0.271296847205
                        data = (data - mean) / std
                    X_list.insert(params['vlist'].index('adt'), data)  # 插入到原vlist指定位置
            
            if 'h' in params['vlist']:
                h_data = np.load("D:/subsurface marine heatwaves data/water_depth_large.npz")['water_depth'][np.newaxis, np.newaxis]  # [1, 1, X, Y]
                data = np.tile(h_data, (1, end-start, 1, 1))  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = 3022.16841881, 2348.80965868
                    data = (data - mean) / std
                X_list.insert(params['vlist'].index('h'), data)
            
            if 'u10' in params['vlist']:
                wind_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/ERA5/era5_daily_2010-2023.nc")
                data = wind_file.variables['u10_day'][start:end].values[np.newaxis]  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = -1.54994392047, 4.30704358882
                    data = (data - mean) / std
                data = np.flip(data,2)
                X_list.insert(params['vlist'].index('u10'), data)
            
            if 'v10' in params['vlist']:
                wind_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/ERA5/era5_daily_2010-2023.nc")
                data = wind_file.variables['v10_day'][start:end].values[np.newaxis]  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = -0.443924070413, 3.71276861017
                    data = (data - mean) / std
                data = np.flip(data,2)
                X_list.insert(params['vlist'].index('v10'), data)
    
            if 'oisst' in params['vlist']:
                oisst_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/OISST/calibrated_sst_from125_2010-2023_new.nc")
                data = oisst_file.variables['calibrated_sst'][start:end].values[np.newaxis]  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = 26.4240595035, 4.71648673939
                    data = (data - mean) / std
                X_list.insert(params['vlist'].index('oisst'), data)
            
            if 'csss' in params['vlist']:
                csss_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/GLORYS_SSS/cmems_glorys_2010-2023_SSS_first_layer.nc")
                data = csss_file.variables['sss_glor'][start:end].values[np.newaxis]  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = 33.8852060589, 1.57568720645
                    data = (data - mean) / std
                X_list.insert(params['vlist'].index('csss'), data)
    
            if 'wind_curl' in params['vlist']:
                wind_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/ERA5/era5_daily_2010-2023.nc")
                u = wind_file.variables['u10_day'][start:end].values[np.newaxis]  # [1, B, X, Y]
                v = wind_file.variables['v10_day'][start:end].values[np.newaxis]  # [1, B, X, Y]
                tau_x, tau_y = calculate_wind_stress(u, v)
                curl = calculate_wind_stress_curl(tau_x, tau_y, wind_file['longitude'].values, wind_file['latitude'].values)
                if params['input_norm']:
                    mean, std = -5.08988200764e-09, 3.46038138626e-07
                    curl = (curl - mean) / std
                curl = np.flip(curl, 2)
                X_list.insert(params['vlist'].index('wind_curl'), curl)
            
            if 'gradoi' in params['vlist']:
                oifile = xr.open_dataset(r'D:/subsurface marine heatwaves data/OISST/calibrated_sst_from125_2010-2023_new.nc')
                dtdx = oifile['calibrated_sst'][start:end].differentiate('longitude')
                dtdy = oifile['calibrated_sst'][start:end].differentiate('latitude')
                data = np.log(dtdx**2 + dtdy**2).values[np.newaxis]
                if params['input_norm']:
                    mean, std = -1.71045694633, 2.09019437044
                    data = (data - mean) / std
                X_list.insert(params['vlist'].index('gradoi'), data)
                
            if 'gradadt' in params['vlist']:
                nc_file = xr.open_dataset(r"D:/subsurface marine heatwaves data/ADT/calibrated_adt_from125_2010-2023.nc")
                dtdx = nc_file['calibrated_adt'][start:end].differentiate('longitude')
                dtdy = nc_file['calibrated_adt'][start:end].differentiate('latitude')
                data = np.log(dtdx**2 + dtdy**2).values[np.newaxis]
                if params['input_norm']:
                    mean, std = -5.54760232122, 2.14272967739
                    data = (data - mean) / std
                X_list.insert(params['vlist'].index('gradadt'), data)
            # 3. 合并特征 [B, 1, X, Y, F]
            X = np.concatenate(X_list, axis=0)  # [F, B, X, Y]
            X = np.transpose(X, (1, 2, 3, 0))   # [B, X, Y, F]
            X = X[:, np.newaxis, ...]           # [B, 1, X, Y, F]
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            X[X > 100] = 0  # 特殊值处理
            np.save(os.path.join(savepath, X_npy), X.astype(np.float32))
            
        if os.path.exists(os.path.join(savepath, Y_npy)):
            #print('loading local Y.npy')
            Y = np.load(os.path.join(savepath, Y_npy)).astype(np.float32)
        else:
            # 4. 构建输出Y
            with nc.Dataset(file_path, 'r') as nc_file:
                if params['target'] == 'temp3d':
                    Y = np.stack([nc_file.variables['thetao'+vname_postfix][start:end, l].data 
                                 for l in Level], axis=1)  # [B, L, X, Y]
                    Y = Y[:, np.newaxis]  # [B, 1, L, X, Y]
                    Y = np.transpose(Y, (0, 1, 3, 4, 2))  # [B, 1, X, Y, L]
                    if params['output_norm']:
                        mean = params['thetao_global_mean'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                        std = params['thetao_global_std'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                        Y = (Y - mean)/std
                elif params['target'] == 'salt3d':
                    Y = np.stack([nc_file.variables['so'+vname_postfix][start:end, l].data 
                                 for l in Level], axis=1)  # [B, L, X, Y]
                    Y = Y[:, np.newaxis]  # [B, 1, L, X, Y]
                    Y = np.transpose(Y, (0, 1, 3, 4, 2))  # [B, 1, X, Y, L]
                    if params['output_norm']:
                        mean = params['so_global_mean'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                        std = params['so_global_std'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                        Y = (Y - mean)/std
                        
                elif params['target'] == 'temp':
                    Y = nc_file.variables['thetao'+vname_postfix][start:end, Level].data  # [B, X, Y]
                    Y = Y[:, np.newaxis, :, :, np.newaxis]  # [B, 1, X, Y, 1]
                    
                elif params['target'] == 'salt':
                    Y = nc_file.variables['so'+vname_postfix][start:end, Level].data  # [B, X, Y]
                    Y = Y[:, np.newaxis, :, :, np.newaxis]  # [B, 1, X, Y, 1]
                
                elif params['target'] == 'ts':
                    temp = nc_file.variables['thetao'+vname_postfix][start:end, Level].data  # [B, X, Y]
                    salt = nc_file.variables['so'+vname_postfix][start:end, Level].data  # [B, X, Y]
                    Y = np.stack([temp, salt], axis=-1)  # [B, X, Y, 2]
                    Y = Y[:, np.newaxis]  # [B, 1, X, Y, 2]
            
            Y = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
            Y[Y > 1e10] = 0
            np.save(os.path.join(savepath, Y_npy), Y.astype(np.float32))
        return torch.from_numpy(X[:,:,:]).float(), torch.from_numpy(Y[:,:,:]).float()
    
def calculate_wind_stress(u10, v10, air_density=1.225, drag_coeff=0.0012):
    wind_speed = np.sqrt(u10**2 + v10**2)

    tau_x = air_density * drag_coeff * wind_speed * u10
    tau_y = air_density * drag_coeff * wind_speed * v10
    
    return tau_x, tau_y

def calculate_wind_stress_curl(tau_x, tau_y, lon, lat):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    R = 6371000
    
    dtauy_dx, dtauy_dy = np.gradient(tau_y, axis=(3, 2))
    dtaux_dx, dtaux_dy = np.gradient(tau_x, axis=(3, 2))
    
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    
    dx = R * np.cos(lat_rad[:, np.newaxis]) * dlon
    dy = R * dlat
    
    curl = dtauy_dx/dx - dtaux_dy/dy
    
    return curl


def GetTestData(params, time_range=None, 
                adt_name=r"C:\Github\DORS\Data\ADT\calibrated_adt_from125_2010-2024.nc",
                sst_name=r"C:\Github\DORS\Data\OISST\calibrated_sst_from125_2010-2024_new.nc"):
    """
    读取特定时间段的数据
    
    Args:
        fname_year: 文件名年份标识
        params: 参数配置字典
        time_range: 时间段筛选条件，格式为 [start_time, end_time]
    
    Returns:
        X, Y: 输入和输出数据
    """
    vname_postfix = '_glor'
    Level = np.array([4,7,11,14,16,18,21,24,25,26,27,30,32,34,37,39,42,44,46]) #5-1000m
    file_path = r"C:\Github\DORS\Data\GLORYS\cmems_glorys_2010-2024.nc"
    with xr.open_dataset(file_path) as ds:
        lon0 = ds['longitude'].values
        lat0 = ds['latitude'].values
        #lon0, lat0 = np.meshgrid(lon, lat)
    # 1. 初始化特征列表
    X_list = []
    
    with xr.open_dataset(file_path) as ds:
        start = 0
        end = len(ds.time.sel(time=slice(time_range[0], time_range[1])))
        # 按vlist顺序读取并处理每个变量
        for var in params['vlist']:
            if var == 'lon':
                lon = ds['longitude'].values
                lat = ds['latitude'].values
                lon_grid, _ = np.meshgrid(lon, lat)
                data = np.broadcast_to(lon_grid[np.newaxis, np.newaxis], 
                                     (1, end-start, *lon_grid.shape))  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = 132.5, 15.9491379077
                    data = (data - mean) / std
                X_list.append(data[:,:,-121:])
            
            elif var == 'lat':
                lon = ds['longitude'].values
                lat = ds['latitude'].values
                _, lat_grid = np.meshgrid(lon, lat)
                data = np.broadcast_to(lat_grid[np.newaxis, np.newaxis], 
                                     (1, end-start, *lat_grid.shape))  # [1, B, X, Y]
                if params['input_norm']:
                    mean, std = 20, 11.6189500386
                    data = (data - mean) / std
                X_list.append(data[:,:,-121:])
                        
    # 2. 读取外部数据
    if 'time_sin' in params['vlist']:
        with xr.open_dataset(file_path) as ds:
            if time_range is not None:
                ds = ds.sel(time=slice(time_range[0], time_range[1]))
            time_vals = np.sin(ds.time.dt.dayofyear.values / 365 * 2 * np.pi)
            data = np.broadcast_to(time_vals[np.newaxis, :, np.newaxis, np.newaxis], 
                                       (1, end-start, lon_grid.shape[0], lon_grid.shape[1]))  # [1, B, X, Y]
            X_list.insert(params['vlist'].index('time_sin'), data[:,:,-121:])
            
    if 'time_cos' in params['vlist']:
        with xr.open_dataset(file_path) as ds:
            if time_range is not None:
                ds = ds.sel(time=slice(time_range[0], time_range[1]))
            time_vals = np.cos(ds.time.dt.dayofyear.values / 365 * 2 * np.pi)
            data = np.broadcast_to(time_vals[np.newaxis, :, np.newaxis, np.newaxis], 
                                       (1, end-start, lon_grid.shape[0], lon_grid.shape[1]))  # [1, B, X, Y]
            X_list.insert(params['vlist'].index('time_cos'), data[:,:,-121:])
        
    if 'adt' in params['vlist']:
        with xr.open_dataset(adt_name) as nc_file:
            if time_range is not None:
                nc_file = nc_file.sel(time=slice(time_range[0], time_range[1]))
            #如果为AVISO
            #data = nc_file['calibrated_adt'][start:end,40:].values[np.newaxis]            # [1, B, X, Y]
            #如果为和江南预测
            data = nc_file['calibrated_adt'][start:end,:,40:].values[np.newaxis].swapaxes(2,3)   # [1, B, X, Y]
            if params['input_norm']:
                mean, std = 0.740288219647, 0.271296847205
                data = (data - mean) / std
            X_list.insert(params['vlist'].index('adt'), data[:,:,-121:])
    
    if 'h' in params['vlist']:
        h_data = np.load("D:/subsurface marine heatwaves data/water_depth_large.npz")['water_depth'][np.newaxis, np.newaxis]  # [1, 1, X, Y]
        data = np.tile(h_data, (1, end-start, 1, 1))  # [1, B, X, Y]
        if params['input_norm']:
            mean, std = 3022.16841881, 2348.80965868
            data = (data - mean) / std
        X_list.insert(params['vlist'].index('h'), data[:,:,-121:])
    
    if 'u10' in params['vlist']:
        wind_file = xr.open_dataset(r"C:\Github\DORS\Data\ERA5\era5_daily_2010-2024.nc")
        if time_range is not None:
            wind_file = wind_file.sel(valid_time=slice(time_range[0], time_range[1]))
        data = wind_file['u10_day'].values[start:end][np.newaxis]  # [1, B, X, Y]
        if params['input_norm']:
            mean, std = -1.54994392047, 4.30704358882
            data = (data - mean) / std
        data = np.flip(data, 2)
        X_list.insert(params['vlist'].index('u10'), data[:,:,-121:])
    
    if 'v10' in params['vlist']:
        wind_file = xr.open_dataset(r"C:\Github\DORS\Data\ERA5\era5_daily_2010-2024.nc")
        if time_range is not None:
            wind_file = wind_file.sel(valid_time=slice(time_range[0], time_range[1]))
        data = wind_file['v10_day'].values[start:end][np.newaxis]  # [1, B, X, Y]
        if params['input_norm']:
            mean, std = -0.443924070413, 3.71276861017
            data = (data - mean) / std
        data = np.flip(data, 2)
        X_list.insert(params['vlist'].index('v10'), data[:,:,-121:])

    if 'oisst' in params['vlist']:
        oisst_file = xr.open_dataset(sst_name)
        if time_range is not None:
            oisst_file = oisst_file.sel(time=slice(time_range[0], time_range[1]))
        #data = oisst_file['calibrated_sst'].values[start:end][np.newaxis]  # [1, B, X, Y]
        data = oisst_file['predicted_sst'].interp(lon=lon0,lat=lat0).values[start:end][np.newaxis]  # [1, B, X, Y]
        if params['input_norm']:
            mean, std = 26.4240595035, 4.71648673939
            data = (data - mean) / std
        X_list.insert(params['vlist'].index('oisst'), data[:,:,-121:])
    
    if 'csss' in params['vlist']:
        csss_file = xr.open_dataset(r"C:\Github\DORS\Data\SMAP\sss_glor_from_Copernicus_my[-2023-12-16]_nrt[2023-12-17-2024].nc")
        if time_range is not None:
            csss_file = csss_file.sel(time=slice(time_range[0], time_range[1]))
        data = csss_file['sss_glor'].values[start:end][np.newaxis]  # [1, B, X, Y]
        if params['input_norm']:
            mean, std = 33.8852060589, 1.57568720645
            data = (data - mean) / std
        X_list.insert(params['vlist'].index('csss'), data[:,:,-121:])

    if 'wind_curl' in params['vlist']:
        wind_file = xr.open_dataset(r"C:\Github\DORS\Data\ERA5\era5_daily_2010-2024.nc")
        if time_range is not None:
            wind_file = wind_file.sel(valid_time=slice(time_range[0], time_range[1]))
        u = wind_file['u10_day'].values[start:end][np.newaxis]  # [1, B, X, Y]
        v = wind_file['v10_day'].values[start:end][np.newaxis]  # [1, B, X, Y]
        tau_x, tau_y = calculate_wind_stress(u, v)
        curl = calculate_wind_stress_curl(tau_x, tau_y, wind_file['longitude'].values, wind_file['latitude'].values)
        if params['input_norm']:
            mean, std = -5.08988200764e-09, 3.46038138626e-07
            curl = (curl - mean) / std
        curl = np.flip(curl, 2)
        X_list.insert(params['vlist'].index('wind_curl'), curl[:,:,-121:])
    
    if 'gradoi' in params['vlist']:
        oifile = xr.open_dataset(r'C:\Github\DORS\Data\OISST\calibrated_sst_from125_2010-2024_new.nc')
        if time_range is not None:
            oifile = oifile.sel(time=slice(time_range[0], time_range[1]))
        dtdx = oifile['calibrated_sst'][start:end].differentiate('longitude')
        dtdy = oifile['calibrated_sst'][start:end].differentiate('latitude')
        data = np.log(dtdx**2 + dtdy**2).values[np.newaxis]
        if params['input_norm']:
            mean, std = -1.71045694633, 2.09019437044
            data = (data - mean) / std
        X_list.insert(params['vlist'].index('gradoi'), data[:,:,-121:])
        
    if 'gradadt' in params['vlist']:
        nc_file = xr.open_dataset(adt_name)
        if time_range is not None:
            nc_file = nc_file.sel(time=slice(time_range[0], time_range[1]))
        
        #如果是AVISO
        adt = nc_file['calibrated_adt'][start:end,40:]            # [1, B, X, Y]
        dtdx = adt.differentiate('longitude')
        dtdy = adt.differentiate('latitude')        
        data = np.log(dtdx**2 + dtdy**2).values[np.newaxis]       # [1, B, X, Y]
        #如果何江南预测
        adt = nc_file['calibrated_adt'][start:end,:,40:]            # [1, B, X, Y]
        dtdx = adt.differentiate('longitude')
        dtdy = adt.differentiate('latitude')        
        data = np.log(dtdx**2 + dtdy**2).values[np.newaxis].swapaxes(2,3)       # [1, B, X, Y]
        if params['input_norm']:
            mean, std = -5.54760232122, 2.14272967739
            data = (data - mean) / std
        X_list.insert(params['vlist'].index('gradadt'), data[:,:,-121:])
    
    # 3. 合并特征 [B, 1, X, Y, F]
    X = np.concatenate(X_list, axis=0)  # [F, B, X, Y]
    X = np.transpose(X, (1, 2, 3, 0))   # [B, X, Y, F]
    X = X[:, np.newaxis, ...]           # [B, 1, X, Y, F]
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X[X > 100] = 0  # 特殊值处理
        
    # 4. 构建输出Y
    with xr.open_dataset(file_path) as ds:
        if time_range is not None:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))
            
        if params['target'] == 'temp3d':
            Y = np.stack([ds['thetao' + vname_postfix].isel(depth=l).values[start:end] 
                         for l in Level], axis=1)  # [B, L, X, Y]
            Y = Y[:, np.newaxis]  # [B, 1, L, X, Y]
            Y = np.transpose(Y, (0, 1, 3, 4, 2))  # [B, 1, X, Y, L]
            if params['output_norm']:
                mean = params['thetao_global_mean'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                std = params['thetao_global_std'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                Y = (Y - mean)/std
        elif params['target'] == 'salt3d':
            Y = np.stack([ds['so' + vname_postfix].isel(depth=l).values[start:end] 
                         for l in Level], axis=1)  # [B, L, X, Y]
            Y = Y[:, np.newaxis]  # [B, 1, L, X, Y]
            Y = np.transpose(Y, (0, 1, 3, 4, 2))  # [B, 1, X, Y, L]
            if params['output_norm']:
                mean = params['so_global_mean'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                std = params['so_global_std'][np.newaxis,np.newaxis,np.newaxis,np.newaxis]
                Y = (Y - mean)/std
                
        elif params['target'] == 'temp':
            Y = ds['thetao' + vname_postfix].isel(depth=Level).values[start:end]  # [B, X, Y]
            Y = Y[:, np.newaxis, :, :, np.newaxis]  # [B, 1, X, Y, 1]
            
        elif params['target'] == 'salt':
            Y = ds['so' + vname_postfix].isel(depth=Level).values[start:end]  # [B, X, Y]
            Y = Y[:, np.newaxis, :, :, np.newaxis]  # [B, 1, X, Y, 1]
        
        elif params['target'] == 'ts':
            temp = ds['thetao' + vname_postfix].isel(depth=Level).values[start:end]  # [B, X, Y]
            salt = ds['so' + vname_postfix].isel(depth=Level).values[start:end]  # [B, X, Y]
            Y = np.stack([temp, salt], axis=-1)  # [B, X, Y, 2]
            Y = Y[:, np.newaxis]  # [B, 1, X, Y, 2]
    
    Y = np.nan_to_num(Y, nan=0, posinf=0, neginf=0)
    Y[Y > 1e10] = 0
    
    return torch.from_numpy(X).float(), torch.from_numpy(Y[:,:,-121:]).float()



def crps_loss(predicted, observed):
    """
    优化的CRPS实现，更节省内存
    """
    B, T, X, Y, Z = predicted.shape
    n_members = T * X * Y * Z
    
    # 扁平化处理
    pred_flat = predicted.view(B, n_members)
    obs_flat = observed.view(B, -1)
    
    # 排序预测值
    pred_sorted, _ = torch.sort(pred_flat, dim=1)
    
    # 计算CRPS
    alpha = torch.arange(1, n_members + 1, device=predicted.device).float() / n_members
    beta = torch.arange(0, n_members, device=predicted.device).float() / n_members
    
    # 找到观测值在排序后预测值中的位置
    idx = torch.searchsorted(pred_sorted, obs_flat)
    idx = torch.clamp(idx, 0, n_members - 1)
    
    # 计算CRPS
    term1 = (obs_flat - pred_sorted).abs().mean(dim=1)
    term2 = (pred_sorted * (2 * alpha - 1)).mean(dim=1)
    
    crps = term1 - term2
    
    return crps.sum()

class MSE_CRPS_Loss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        """
        结合MSE和CRPS的损失函数
        
        Args:
            alpha: MSE损失的权重，CRPS损失的权重为(1-alpha)
            reduction: 损失缩减方式，'mean'或'sum'
        """
        super(MSE_CRPS_Loss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, input, target, mask=None):
        """
        Args:
            input: 预测值，形状为(batch_size, ...)
            target: 真实值，形状与input相同
        """
        # 计算MSE损失
        if mask is not None:
            mse_loss = self.mse_loss(input, target)
        else:
            mse_loss = self.mse_loss(input[mask], target[mask])
        
        # 计算CRPS损失
        crps = crps_loss(input, target)
        
        # 结合两种损失
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * crps
        
        return combined_loss
    