from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import Normalize  # 用于统一色标范围
from matplotlib.cm import ScalarMappable

def calculate_metrics(pred, obs):
    """计算MAE, MRE, CC"""
    valid_mask = ~np.isnan(pred) & ~np.isnan(obs)
    if np.sum(valid_mask) < 2:
        return np.nan, np.nan, np.nan
    
    pred_valid, obs_valid = pred[valid_mask], obs[valid_mask]
    rmse = np.mean((pred_valid - obs_valid)**2)**0.5
    obs_no_zero = np.where(obs_valid == 0, 1e-6, obs_valid)
    mre = np.mean(np.abs((pred_valid - obs_valid) / obs_no_zero)) * 100
    cc = np.corrcoef(pred_valid, obs_valid)[0, 1]
    
    return rmse, mre, cc

def plot_temp(lon, lat, temp1, temp2, time, z=0, plot_argo=False, des=''):
    # 转换时间格式
    if isinstance(time, np.datetime64):
        time_str = np.datetime_as_string(time, unit="D").replace("-", "")
    else:
        time_str = time.strftime("%Y%m%d")
    
    if 1==0:
        # 读取 ARGO 数据
        argo_path = f'Data/Argo/argo_{time_str}.nc'
        argo = xr.open_dataset(argo_path)
        
        # 找到最接近 z 的深度层
        depth_idx = np.abs(argo.DEPTH.values - z).argmin()
        argo_z = argo.isel(DEPTH=depth_idx)  # 提取该深度的数据
    
    # 计算所有温度数据的统一色标范围（temp1, temp2, ARGO）
    vmin = 5#min(np.nanmin(temp1), np.nanmin(temp2))
    vmax = 10#max(np.nanmax(temp1), np.nanmax(temp2))
    
    norm = Normalize(vmin=vmin, vmax=vmax)  # 统一标准化
    
    # 创建图形
    fig = figure(figsize=(5, 5), dpi=150)
    
    # 共享的 colorbar（放在右侧）
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    colorbar(sm, cax=cax, label=f'Temperature at {z:.1f}m (°C)')
    
    for i, (temp, title) in enumerate(zip([temp1, temp2], ['Truth', 'Pred']), 1):
        ax = fig.add_subplot(2, 1, i, projection=ccrs.PlateCarree())
        
        # 添加地理特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.add_feature(cfeature.OCEAN, color='lightblue')
        
        # 绘制温度场（使用统一 norm）
        pcm = ax.pcolormesh(lon, lat, temp, cmap='jet', norm=norm, transform=ccrs.PlateCarree())
        
        if plot_argo: # 绘制 ARGO 散点（使用相同 norm 和 cmap）
            sc = ax.scatter(
                argo_z.LONGITUDE, 
                argo_z.LATITUDE, 
                c=argo_z.TEMP, 
                cmap='jet', 
                norm=norm,  # 关键：使用相同的标准化
                s=50, 
                edgecolor='w', 
                transform=ccrs.PlateCarree(),
                label='ARGO'
            )
            
            # 修改文本标注方式
            valid_mask = ~np.isnan(argo_z.TEMP)
            if np.any(valid_mask):
                # 清除现有文本的正确方法
                for txt in ax.texts:
                    txt.remove()
                
                # 添加新文本
                for lon_pt, lat_pt, val in zip(argo_z.LONGITUDE[valid_mask], 
                                             argo_z.LATITUDE[valid_mask], 
                                             argo_z.TEMP[valid_mask]):
                    ax.text(lon_pt + 0.2, lat_pt + 0.2, f"{val:.2f}",
                           fontsize=16, color='black',
                           transform=ccrs.PlateCarree(),
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.legend(loc='upper right')
        
        ax.set_title(f"{title}")
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = gl.right_labels = False
        if i == 1:
            gl.bottom_labels = False
        
        #axis((120,130,20,30))
    suptitle(f'{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} | depth: {z:.2f}m | {des}')
    #tight_layout(rect=[0, 0, 0.9, 1])  # 给 colorbar 留空间
    #savefig(f'{out_dir}/t{z:.1f}.png', dpi=150, bbox_inches='tight')
    show()
    
def plot_vec(x,y,truth, pred, date, index = 0, layer = 0):
    """
    可视化预测结果
    参数:
        forecast_results: 预测结果字典
        current_date: 当前日期
    """
    print("生成可视化图表...")
    
    # 示例: 可视化表层流场预测
    #surface_level = min(forecast_results.keys())  # 最浅层
    # 获取第一天的预测
    u_pred = pred[index,0,:,:,0]
    v_pred = pred[index,0,:,:,1]
    u_true = truth[index,0,:,:,0]
    v_true = truth[index,0,:,:,1]
    
    # 计算流速大小
    speed = np.sqrt(u_pred**2 + v_pred**2)
    
    # 对于多样本数据，我们只有一个预测来比较
    
    # 使用我们拥有的单个预测
    
    fig3 = plt.figure(figsize=(15, 8))
    
    # 创建两个子图进行比较
    ax1 = fig3.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax2 = fig3.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    
    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')
        ax.add_feature(cfeature.COASTLINE)
        #ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        #ax.gridlines(draw_labels={'bottom': True, 'left': True, 'top': False, 'right': False})
        ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=ccrs.PlateCarree())

    print(f"矢量场可视化使用深度层 {layer}")
    
    # 为更好的可视化分别绘制真实场和预测场
    subsample = 2
    
    # 绘制真实值
    ax1.quiver(x[::subsample, ::subsample], y[::subsample, ::subsample], 
               u_true[::subsample, ::subsample], v_true[::subsample, ::subsample], 
               color='red', scale=30)
    ax1.set_title(f'Truth @{date} {layer:.0f}m')
    
    # 绘制预测值
    ax2.quiver(x[::subsample, ::subsample], y[::subsample, ::subsample], 
               u_pred[::subsample, ::subsample], v_pred[::subsample, ::subsample], 
               color='blue', scale=30)
    ax2.set_title(f'Pred ({layer:.0f}m)')
    
    #print(f"显示矢量对比图...")
    plt.show()

def plot_salt(lon, lat, salt1, salt2, time, z=0, plot_argo=False,  des=''):
    # 转换时间格式
    if isinstance(time, np.datetime64):
        time_str = np.datetime_as_string(time, unit="D").replace("-", "")
    else:
        time_str = time.strftime("%Y%m%d")
    
    # 读取 ARGO 数据
    argo_path = f'Data/Argo/argo_{time_str}.nc'
    argo = xr.open_dataset(argo_path)
    
    # 找到最接近 z 的深度层
    depth_idx = np.abs(argo.DEPTH.values - z).argmin()
    argo_z = argo.isel(DEPTH=depth_idx)  # 提取该深度的数据
    
    # 计算所有温度数据的统一色标范围（temp1, temp2, ARGO）
    vmin = min(np.nanmin(salt1), np.nanmin(salt2), np.nanmin(argo_z.PSAL))
    vmax = max(np.nanmax(salt1), np.nanmax(salt2), np.nanmax(argo_z.PSAL))
    vmin = 32;vmax = 35
    norm = Normalize(vmin=vmin, vmax=vmax)  # 统一标准化
    
    # 创建图形
    fig = figure(figsize=(5, 5), dpi=150)
    
    # 共享的 colorbar（放在右侧）
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    colorbar(sm, cax=cax, label=f'Salinity at {z:.1f}m (PSU)')
    
    for i, (salt, title) in enumerate(zip([salt1, salt2], ['Truth', 'Pred']), 1):
        ax = fig.add_subplot(2, 1, i, projection=ccrs.PlateCarree())
        
        # 添加地理特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.add_feature(cfeature.OCEAN, color='lightblue')
        
        # 绘制温度场（使用统一 norm）
        pcm = ax.pcolormesh(lon, lat, salt, cmap='jet', norm=norm, transform=ccrs.PlateCarree())
        if plot_argo:
            # 绘制 ARGO 散点（使用相同 norm 和 cmap）
            sc = ax.scatter(
                argo_z.LONGITUDE, 
                argo_z.LATITUDE, 
                c=argo_z.PSAL, 
                cmap='jet', 
                norm=norm,  # 关键：使用相同的标准化
                s=50, 
                edgecolor='w', 
                transform=ccrs.PlateCarree(),
                label='ARGO'
            )
            
            # 修改文本标注方式
            valid_mask = ~np.isnan(argo_z.PSAL)
            if np.any(valid_mask):
                # 清除现有文本的正确方法
                for txt in ax.texts:
                    txt.remove()
                
                # 添加新文本
                for lon_pt, lat_pt, val in zip(argo_z.LONGITUDE[valid_mask], 
                                             argo_z.LATITUDE[valid_mask], 
                                             argo_z.PSAL[valid_mask]):
                    ax.text(lon_pt + 0.2, lat_pt + 0.2, f"{val:.2f}",
                           fontsize=8, color='black',
                           transform=ccrs.PlateCarree(),
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
            ax.legend(loc='upper right')
        
        ax.set_title(f"{title}")
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = gl.right_labels = False
        if i == 1:
            gl.bottom_labels = False
    
    suptitle(f'{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]} | depth: {z:.2f}m | {des}')
    #tight_layout(rect=[0, 0, 0.9, 1])  # 给 colorbar 留空间
    #savefig(f'{out_dir}/s{z:.1f}.png', dpi=150, bbox_inches='tight')
    show()
    
def plot_metrics(x, y, rmse_spatial, r2_spatial, layer=0):
    """
    可视化空间RMSE和R²分布
    参数:
        x, y: 经纬度网格
        rmse_spatial: 空间RMSE矩阵 [H, W]
        r2_spatial: 空间R²矩阵 [H, W]
        layer: 深度层级
    """
    print(f"生成空间指标可视化图表 (深度层 {layer}m)...")
    
    fig = plt.figure(figsize=(15, 8))
    
    # 创建两个子图
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    
    # 设置地图特征
    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')
        ax.add_feature(cfeature.COASTLINE)
        ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=ccrs.PlateCarree())
    
    # 绘制RMSE空间分布
    rmse_plot = ax1.pcolormesh(x, y, rmse_spatial, 
                              transform=ccrs.PlateCarree(),
                              cmap='viridis', shading='auto')
    ax1.set_title(f'RMSE Spatial Distribution ({layer:.0f}m)')
    fig.colorbar(rmse_plot, ax=ax1, label='RMSE (m/s)')
    
    # 绘制R²空间分布
    r2_plot = ax2.pcolormesh(x, y, r2_spatial,
                            transform=ccrs.PlateCarree(),
                            cmap='coolwarm', vmin=0, vmax=1,
                            shading='auto')
    ax2.set_title(f'R² Spatial Distribution ({layer:.0f}m)')
    fig.colorbar(r2_plot, ax=ax2, label='R²')
    
    plt.tight_layout()
    plt.show()


from matplotlib.pyplot import *
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae
    
def plot_fancy_temp_scatters(argo_dict,pred_temp, description='TEMP'):
    # 1. 准备数据容器 - 为每个深度创建列表存储所有站点的模型和观测温度
    for i, date in enumerate(argo_dict):
        depth_levels = argo_dict[date]['DEPTH'].values
    depth_data = {depth: {'model': [], 'argo': []} for depth in depth_levels}

    # 2. 合并所有时间步的数据
    for date in pred_temp.keys():
        if date not in argo_dict:  # 确保有对应的Argo数据
            continue
        
        # 获取当前日期的数据
        m_temp = pred_temp[date]  # 模型温度 [站点, 深度]
        a_temp = argo_dict[date]['TEMP'].values  # Argo观测温度 [站点, 深度]
        
        # 检查维度是否匹配
        if m_temp.shape != a_temp.shape:
            print(f"维度不匹配 {date}: 模型{m_temp.shape} vs Argo{a_temp.shape}")
            continue
        
        # 为每个深度添加数据
        for depth_idx, depth in enumerate(depth_levels):
            depth_data[depth]['model'].extend(m_temp[:, depth_idx])
            depth_data[depth]['argo'].extend(a_temp[:, depth_idx])

    # 3. 绘制散点图 - 每个深度一个子图
    n_depths = len(depth_levels)
    n_cols = 3  # 每行3个子图
    n_rows = int(np.ceil(n_depths / n_cols))
    
    rmse_all = np.zeros((n_depths,))
    r2_all = np.zeros((n_depths,))
    mae_all = np.zeros((n_depths,))
    
    figure(figsize=(15, 5*n_rows))
    for idx, depth in enumerate(depth_levels, 1):
        subplot(n_rows, n_cols, idx)
        
        model_vals = np.array(depth_data[depth]['model'])
        argo_vals = np.array(depth_data[depth]['argo'])
        
        # 计算评估指标
        rmse, r2, mae = calculate_metrics(argo_vals, model_vals)
        rmse_all[idx-1] = rmse
        r2_all[idx-1] = r2
        mae_all[idx-1] = mae
        
        # 绘制散点图
        scatter(argo_vals, model_vals, alpha=0.5, s=10, label=f'Depth: {depth}m')
        
        # 添加1:1参考线
        min_val = min(np.nanmin(model_vals), np.nanmin(argo_vals))
        max_val = max(np.nanmax(model_vals), np.nanmax(argo_vals))
        plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加指标文本
        text(0.05, 0.85, 
             f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(model_vals[~np.isnan(model_vals) & ~np.isnan(argo_vals)])}',
             transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        xlabel('Argo Observed Temperature (°C)')
        ylabel('Model Temperature (°C)')
        title(f'Depth: {depth}m', fontsize=font_size)
        suptitle(f'{description}', fontsize=font_size)
        grid(True)
        legend()

    tight_layout()
    show()

    # 5. 绘制所有深度的合并散点图
    figure(figsize=(8, 8))
    all_model = []
    all_argo = []
    for depth in depth_levels:
        all_model.extend(depth_data[depth]['model'])
        all_argo.extend(depth_data[depth]['argo'])

    all_model = np.array(all_model)
    all_argo = np.array(all_argo)

    # 计算整体指标
    rmse, r2, mae = calculate_metrics(all_argo, all_model)

    mask = ~np.isnan(all_model) & ~np.isnan(all_argo)
    scatter(all_argo[mask], all_model[mask], alpha=0.3, s=10, c='b', label='All Depths')

    # 添加1:1参考线
    min_val = min(np.nanmin(all_model), np.nanmin(all_argo))
    max_val = max(np.nanmax(all_model), np.nanmax(all_argo))
    plot([min_val, max_val], [min_val, max_val], 'r--')

    # 添加整体指标文本
    text(0.05, 0.85, 
         f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(all_model[mask])}',
         transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    xlabel('Argo Observed Temperature (°C)')
    ylabel('Model Temperature (°C)')
    title(f'{description} All Depths Combined')
    grid(True)
    legend()
    show()
    
    return rmse_all, r2_all, mae_all

def plot_fancy_salt_scatters(argo_dict,pred_salt, description = 'Salt'):
    # 1. 准备数据容器 - 为每个深度创建列表存储所有站点的模型和观测温度
    for i, date in enumerate(argo_dict):
        depth_levels = argo_dict[date]['DEPTH'].values
    depth_data = {depth: {'model': [], 'argo': []} for depth in depth_levels}

    # 2. 合并所有时间步的数据
    for date in pred_salt.keys():
        if date not in argo_dict:  # 确保有对应的Argo数据
            continue
        
        # 获取当前日期的数据
        m_temp = pred_salt[date]  # 模型温度 [站点, 深度]
        a_temp = argo_dict[date]['PSAL'].values  # Argo观测温度 [站点, 深度]
        
        # 检查维度是否匹配
        if m_temp.shape != a_temp.shape:
            print(f"维度不匹配 {date}: 模型{m_temp.shape} vs Argo{a_temp.shape}")
            continue
        
        # 为每个深度添加数据
        for depth_idx, depth in enumerate(depth_levels):
            depth_data[depth]['model'].extend(m_temp[:, depth_idx])
            depth_data[depth]['argo'].extend(a_temp[:, depth_idx])

    # 3. 绘制散点图 - 每个深度一个子图
    n_depths = len(depth_levels)
    n_cols = 3  # 每行3个子图
    n_rows = int(np.ceil(n_depths / n_cols))

    rmse_all = np.zeros((n_depths,))
    r2_all = np.zeros((n_depths,))
    mae_all = np.zeros((n_depths,))
    
    figure(figsize=(15, 5*n_rows))
    for idx, depth in enumerate(depth_levels, 1):
        subplot(n_rows, n_cols, idx)
        
        model_vals = np.array(depth_data[depth]['model'])
        argo_vals = np.array(depth_data[depth]['argo'])
        
        # 计算评估指标
        rmse, r2, mae = calculate_metrics(argo_vals, model_vals)
        rmse_all[idx-1] = rmse
        r2_all[idx-1] = r2
        mae_all[idx-1] = mae
        
        # 绘制散点图
        scatter(argo_vals, model_vals, alpha=0.5, s=10, label=f'Depth: {depth}m')
        
        # 添加1:1参考线
        min_val = 30
        max_val = 35
        plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加指标文本
        text(0.05, 0.85, 
             f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(model_vals[~np.isnan(model_vals) & ~np.isnan(argo_vals)])}',
             transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        xlim((min_val,max_val))
        ylim((min_val,max_val))
        xlabel('Argo Observed Salinity (PSU)')
        ylabel('Model Salinity (PS)')
        title(f'{description} Depth: {depth}m')
        grid(True)
        legend()

    tight_layout()
    show()

    # 5. 绘制所有深度的合并散点图
    figure(figsize=(8, 8))
    all_model = []
    all_argo = []
    for depth in depth_levels:
        all_model.extend(depth_data[depth]['model'])
        all_argo.extend(depth_data[depth]['argo'])

    all_model = np.array(all_model)
    all_argo = np.array(all_argo)

    # 计算整体指标
    rmse, r2, mae = calculate_metrics(all_argo, all_model)

    mask = ~np.isnan(all_model) & ~np.isnan(all_argo)
    scatter(all_argo[mask], all_model[mask], alpha=0.3, s=10, c='b', label='All Depths')

    # 添加1:1参考线
    min_val = 30
    max_val = 35
    
    plot([min_val, max_val], [min_val, max_val], 'r--')

    # 添加整体指标文本
    text(0.05, 0.85, 
         f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(all_model[mask])}',
         transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    xlim((min_val,max_val))
    ylim((min_val,max_val))
    xlabel('Argo Observed Salinity (PSU)')
    ylabel('Model Salinity (PS)')
    title(f'{description} All Depths Combined')
    grid(True)
    legend()
    show()   
    
    return rmse_all, r2_all, mae_all

def plot_fancy_scatter(argo_dict, pred_data, description='TEMP', font_size=16, out_dir = 'figures/'):
    """
    统一的散点图绘制函数，支持温度和盐度数据
    
    参数:
    argo_dict: 包含Argo观测数据的字典
    pred_data: 模型预测数据（温度或盐度）
    description: 数据描述，用于自动判断变量类型
    font_size: 图中所有文字的字号，默认值为12
    """
    
    # 自动判断变量类型
    if 'salt' in description.lower() or 'sal' in description.lower():
        var_type = 'salt'
        argo_var = 'PSAL'  # Argo中的盐度变量名
        units = 'Salinity (PSU)'
        # 盐度的固定范围
        min_val_fixed, max_val_fixed = 30, 36
    else:
        var_type = 'temp'
        argo_var = 'TEMP'  # Argo中的温度变量名
        units = 'Temperature (°C)'
        min_val_fixed, max_val_fixed = None, None  # 温度使用动态范围

    # 1. 准备数据容器 - 为每个深度创建列表存储所有站点的模型和观测数据
    for i, date in enumerate(argo_dict):
        depth_levels = argo_dict[date]['DEPTH'].values
    depth_data = {depth: {'model': [], 'argo': []} for depth in depth_levels}

    # 2. 合并所有时间步的数据
    for date in pred_data.keys():
        if date not in argo_dict:  # 确保有对应的Argo数据
            continue
        
        # 获取当前日期的数据
        m_data = pred_data[date]  # 模型数据 [站点, 深度]
        a_data = argo_dict[date][argo_var].values  # Argo观测数据 [站点, 深度]
        
        # 检查维度是否匹配
        if m_data.shape != a_data.shape:
            print(f"维度不匹配 {date}: 模型{m_data.shape} vs Argo{a_data.shape}")
            continue
        
        # 为每个深度添加数据
        for depth_idx, depth in enumerate(depth_levels):
            depth_data[depth]['model'].extend(m_data[:, depth_idx])
            depth_data[depth]['argo'].extend(a_data[:, depth_idx])

    # 3. 绘制散点图 - 每个深度一个子图
    n_depths = len(depth_levels)
    n_cols = 3  # 每行3个子图
    n_rows = int(np.ceil(n_depths / n_cols))
    
    rmse_all = np.zeros((n_depths,))
    r2_all = np.zeros((n_depths,))
    mae_all = np.zeros((n_depths,))
    
    figure(figsize=(15, 5*n_rows))
    for idx, depth in enumerate(depth_levels, 1):
        subplot(n_rows, n_cols, idx)
        
        model_vals = np.array(depth_data[depth]['model'])
        argo_vals = np.array(depth_data[depth]['argo'])
        
        # 计算评估指标
        rmse, r2, mae = calculate_metrics(argo_vals, model_vals)
        rmse_all[idx-1] = rmse
        r2_all[idx-1] = r2
        mae_all[idx-1] = mae
        
        # 绘制散点图
        scatter(argo_vals, model_vals, alpha=0.5, s=10, label=f'Depth: {depth}m')
        
        # 确定坐标轴范围
        if var_type == 'salt':
            min_val, max_val = min_val_fixed, max_val_fixed
        else:
            min_val = min(np.nanmin(model_vals), np.nanmin(argo_vals))
            max_val = max(np.nanmax(model_vals), np.nanmax(argo_vals))
        
        # 添加1:1参考线
        plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加指标文本
        text(0.05, 0.85, 
             f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(model_vals[~np.isnan(model_vals) & ~np.isnan(argo_vals)])}',
             transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
             fontsize=font_size)
        
        # 设置坐标轴范围和标签
        if var_type == 'salt':
            xlim((min_val, max_val))
            ylim((min_val, max_val))
        
        xlabel(f'Argo Observed {units}', fontsize=font_size)
        ylabel(f'Model {units}', fontsize=font_size)
        title(f'Depth: {depth}m', fontsize=font_size)
        suptitle(f'{description}', fontsize=font_size)
        grid(True)
        legend(fontsize=font_size)
        
        # 设置刻度标签字号
        tick_params(labelsize=font_size)

    tight_layout()
    show()

    # 5. 绘制所有深度的合并散点图
    figure(figsize=(8, 8))
    all_model = []
    all_argo = []
    for depth in depth_levels:
        all_model.extend(depth_data[depth]['model'])
        all_argo.extend(depth_data[depth]['argo'])

    all_model = np.array(all_model)
    all_argo = np.array(all_argo)

    # 计算整体指标
    rmse, r2, mae = calculate_metrics(all_argo, all_model)

    mask = ~np.isnan(all_model) & ~np.isnan(all_argo)
    scatter(all_argo[mask], all_model[mask], alpha=0.3, s=10, c='b', label='All Depths')

    # 确定坐标轴范围
    if var_type == 'salt':
        min_val, max_val = min_val_fixed, max_val_fixed
    else:
        min_val = min(np.nanmin(all_model), np.nanmin(all_argo))
        max_val = max(np.nanmax(all_model), np.nanmax(all_argo))
    
    # 添加1:1参考线
    plot([min_val, max_val], [min_val, max_val], 'r--')

    # 添加整体指标文本
    text(0.05, 0.85, 
         f'RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}\nn={len(all_model[mask])}',
         transform=gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
         fontsize=font_size)
    
    # 设置坐标轴范围和标签
    if var_type == 'salt':
        xlim((min_val, max_val))
        ylim((min_val, max_val))
    
    xlabel(f'Argo Observed {units}', fontsize=font_size)
    ylabel(f'Model {units}', fontsize=font_size)
    title(f'{description} All Depths', fontsize=font_size)
    grid(True)
    legend(fontsize=font_size)
    
    # 设置刻度标签字号
    tick_params(labelsize=font_size)
    
    savefig(f'{out_dir}/{description} All Depths Combined.png', dpi=60)
    show()
    
    return rmse_all, r2_all, mae_all

if __name__=='__main__':
    import datetime
    time = [datetime.datetime(2020,1,1) + datetime.timedelta(days=i) for i in range(1096)]
    import pandas as pd
    
    time = pd.to_datetime(time, format='%Y%m%d')
    
    fname = r"D:\03-WorkingSync\DORS-SST-forecast\dSST_EF_1f_64_mask_Xnorm\dSST_EF_1f_64_mask_Xnorm_epoch99.h5_predictions_sst.nc"
    
    time_list = {'2021-06-22','2021-12-22','2022-06-22','2022-12-22'}
    for tt in time_list:
        tinx = time.get_loc(tt)
        with xr.open_dataset(fname) as ds:
            lon = ds.lon
            lat = ds.lat
            y_pred_all = ds['predicted_sst'][tinx].values - ds['observed_sst'][tinx-1].values 
            y_true_all = ds['observed_sst'][tinx].values - ds['observed_sst'][tinx-1].values
            plot_temp(lon,lat, y_true_all, y_pred_all, time = time[tinx], z = 0,plot_argo=False,des='64-bit')
    
    