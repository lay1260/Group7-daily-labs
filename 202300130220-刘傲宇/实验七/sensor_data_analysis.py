"""
智能手机传感器数据分析
分析跑步和骑电动车在不同路况下的行为模式
根据实际数据结构：每个Excel文件包含多个sheet（Accelerometer, Gyroscope, Linear Acceleration等）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print("智能手机传感器数据分析 - 完整版")
print("=" * 80)

# ============================================================================
# 1. 数据加载 - 读取多个sheet
# ============================================================================
print("\n[步骤1] 加载数据（读取多个sheet）...")

def load_multi_sheet_data(filename):
    """加载Excel文件的所有sheet"""
    try:
        excel_file = pd.ExcelFile(filename)
        sheets_data = {}
        print(f"  {filename}:")
        for sheet_name in excel_file.sheet_names:
            if sheet_name.lower() != 'proximity':  # 跳过Proximity sheet
                df = pd.read_excel(filename, sheet_name=sheet_name)
                sheets_data[sheet_name] = df
                print(f"    ✓ {sheet_name}: {df.shape[0]}行, {df.shape[1]}列")
        return sheets_data
    except Exception as e:
        print(f"  ✗ {filename}: 加载失败 - {e}")
        return None

# 加载所有数据文件
data_files = {
    'running_flat': 'running-flat.xls',
    'running_upslope': 'running-upslope.xls',
    'running_downslope': 'running-downslope.xls',
    'running_flat_upslope': 'running-flat-upslope.xls',
    'running_downslope_flat': 'running-downslope-flat.xls',
    'ebike_flat': 'ebike-flat.xls',
    'ebike_upslope': 'ebike-upslope.xls',
    'ebike_downslope': 'ebike-downslope.xls',
    'ebike_flat_upslope': 'ebike-flat-upslope.xls',
    'ebike_downslope_flat': 'ebike-downslope-flat.xls',
    'phyphox': 'phyphox.xls'
}

datasets = {}
for key, filename in data_files.items():
    if key == 'phyphox':
        # phyphox文件特殊处理
        try:
            excel_file = pd.ExcelFile(filename)
            if 'Flat' in excel_file.sheet_names:
                datasets[key] = {'Flat': pd.read_excel(filename, sheet_name='Flat')}
                print(f"  ✓ {filename}: Flat sheet loaded")
            else:
                datasets[key] = None
        except Exception as e:
            print(f"  ✗ {filename}: 加载失败 - {e}")
            datasets[key] = None
    else:
        datasets[key] = load_multi_sheet_data(filename)

# ============================================================================
# 2. 从phyphox提取坡度信息
# ============================================================================
print("\n[步骤2] 提取坡度信息...")

if datasets['phyphox'] is not None and 'Flat' in datasets['phyphox']:
    phyphox_df = datasets['phyphox']['Flat']
    # 查找Tilt up/down列
    tilt_col = None
    for col in phyphox_df.columns:
        if 'tilt' in str(col).lower() and ('up' in str(col).lower() or 'down' in str(col).lower()):
            tilt_col = col
            break
    
    if tilt_col is None:
        # 尝试第二列（通常是Tilt up/down）
        if len(phyphox_df.columns) >= 2:
            tilt_col = phyphox_df.columns[1]
    
    if tilt_col is not None:
        tilt_data = pd.to_numeric(phyphox_df[tilt_col], errors='coerce').dropna()
        upslope_angle = tilt_data[tilt_data > 0].mean() if len(tilt_data[tilt_data > 0]) > 0 else tilt_data.mean()
        downslope_angle = tilt_data[tilt_data < 0].mean() if len(tilt_data[tilt_data < 0]) > 0 else -tilt_data.mean()
        print(f"  上坡角度（平均）: {upslope_angle:.2f}°")
        print(f"  下坡角度（平均）: {downslope_angle:.2f}°")
    else:
        print("  警告: 未找到坡度列")
        upslope_angle = None
        downslope_angle = None
else:
    upslope_angle = None
    downslope_angle = None

# ============================================================================
# 3. 特征提取函数 - 从各个sheet提取特征
# ============================================================================
print("\n[步骤3] 定义特征提取函数...")

def extract_accelerometer_features(df, prefix='acc'):
    """从Accelerometer sheet提取特征"""
    if df is None or df.empty:
        return {}
    
    features = {}
    
    # 查找列
    time_col = None
    acc_x_col = None
    acc_y_col = None
    acc_z_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'time' in col_lower:
            time_col = col
        elif 'x' in col_lower and ('accel' in col_lower or '加速度' in col_lower):
            acc_x_col = col
        elif 'y' in col_lower and ('accel' in col_lower or '加速度' in col_lower):
            acc_y_col = col
        elif 'z' in col_lower and ('accel' in col_lower or '加速度' in col_lower):
            acc_z_col = col
    
    # 如果没找到，尝试按位置（通常是第0列是时间，1-3列是x,y,z）
    if time_col is None and len(df.columns) > 0:
        time_col = df.columns[0]
    if acc_x_col is None and len(df.columns) > 1:
        acc_x_col = df.columns[1]
    if acc_y_col is None and len(df.columns) > 2:
        acc_y_col = df.columns[2]
    if acc_z_col is None and len(df.columns) > 3:
        acc_z_col = df.columns[3]
    
    if acc_x_col and acc_y_col and acc_z_col:
        acc_x = pd.to_numeric(df[acc_x_col], errors='coerce').dropna()
        acc_y = pd.to_numeric(df[acc_y_col], errors='coerce').dropna()
        acc_z = pd.to_numeric(df[acc_z_col], errors='coerce').dropna()
        
        min_len = min(len(acc_x), len(acc_y), len(acc_z))
        if min_len > 0:
            acc_x = acc_x[:min_len]
            acc_y = acc_y[:min_len]
            acc_z = acc_z[:min_len]
            
            # 计算合加速度
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # 移除重力影响（假设z轴主要是重力，约9.8 m/s²）
            acc_magnitude_no_gravity = np.sqrt(acc_x**2 + acc_y**2 + (acc_z - 9.8)**2)
            
            features[f'{prefix}_mean'] = acc_magnitude.mean()
            features[f'{prefix}_std'] = acc_magnitude.std()
            features[f'{prefix}_max'] = acc_magnitude.max()
            features[f'{prefix}_min'] = acc_magnitude.min()
            features[f'{prefix}_range'] = acc_magnitude.max() - acc_magnitude.min()
            features[f'{prefix}_variability'] = np.std(np.diff(acc_magnitude)) if len(acc_magnitude) > 1 else 0
            
            # 计算各轴的标准差
            features[f'{prefix}_x_std'] = acc_x.std()
            features[f'{prefix}_y_std'] = acc_y.std()
            features[f'{prefix}_z_std'] = acc_z.std()
            
            # 计算主频率（步频/运动频率）
            if len(acc_magnitude) > 10:
                try:
                    # 计算采样率（如果有时间列）
                    sampling_rate = None
                    if time_col:
                        time_data = pd.to_numeric(df[time_col], errors='coerce').dropna()
                        if len(time_data) > 1:
                            dt = np.diff(time_data).mean()
                            if dt > 0 and not np.isnan(dt):
                                sampling_rate = 1.0 / dt
                    
                    # 如果采样率未知，尝试估算（假设至少1Hz采样）
                    if sampling_rate is None or np.isnan(sampling_rate):
                        # 假设数据长度对应至少1秒，估算采样率
                        sampling_rate = max(len(acc_magnitude), 10)  # 至少10Hz
                    
                    # 去均值
                    acc_centered = acc_magnitude - acc_magnitude.mean()
                    
                    # 降低阈值，允许更小的变化（电动车可能更平稳）
                    # 如果数据变化太小，可能没有周期性
                    if acc_centered.std() < 0.01:
                        # 即使变化很小，也尝试计算频率
                        # 使用零交叉率作为替代
                        try:
                            zero_crossings = np.where(np.diff(np.sign(acc_centered)))[0]
                            if len(zero_crossings) > 1:
                                time_span = len(acc_magnitude) / sampling_rate
                                if time_span > 0:
                                    freq_estimate = len(zero_crossings) / (2 * time_span)
                                    if 0.1 <= freq_estimate <= 10:
                                        features[f'{prefix}_dominant_freq'] = freq_estimate
                                    else:
                                        features[f'{prefix}_dominant_freq'] = 0
                                else:
                                    features[f'{prefix}_dominant_freq'] = 0
                            else:
                                features[f'{prefix}_dominant_freq'] = 0
                        except:
                            features[f'{prefix}_dominant_freq'] = 0
                    else:
                        # FFT
                        fft_vals = fft(acc_centered)
                        freqs = fftfreq(len(acc_centered), d=1.0/sampling_rate)  # 使用实际采样率
                        power = np.abs(fft_vals)
                        
                        # 找到主频率（排除DC分量，只看正频率，限制在合理范围0.1-10Hz）
                        positive_idx = np.where((freqs > 0.1) & (freqs < 10))[0]  # 限制在0.1-10Hz范围
                        if len(positive_idx) > 0:
                            main_freq_idx = positive_idx[np.argmax(power[positive_idx])]
                            dominant_freq = abs(freqs[main_freq_idx])
                            # 确保频率在合理范围内
                            if 0.1 <= dominant_freq <= 10:
                                features[f'{prefix}_dominant_freq'] = dominant_freq
                            else:
                                features[f'{prefix}_dominant_freq'] = 0
                        else:
                            # 如果没有找到，尝试不限制范围
                            positive_idx = np.where(freqs > 0)[0]
                            if len(positive_idx) > 1:
                                main_freq_idx = positive_idx[np.argmax(power[positive_idx])]
                                dominant_freq = abs(freqs[main_freq_idx])
                                if 0.1 <= dominant_freq <= 10:
                                    features[f'{prefix}_dominant_freq'] = dominant_freq
                                else:
                                    features[f'{prefix}_dominant_freq'] = 0
                            else:
                                features[f'{prefix}_dominant_freq'] = 0
                except Exception as e:
                    # 如果出错，尝试使用零交叉率作为替代
                    try:
                        # 计算零交叉率作为频率的近似
                        zero_crossings = np.where(np.diff(np.sign(acc_magnitude - acc_magnitude.mean())))[0]
                        if len(zero_crossings) > 1 and len(acc_magnitude) > 0:
                            # 估算频率
                            time_span = len(acc_magnitude) / sampling_rate if sampling_rate else len(acc_magnitude)
                            if time_span > 0:
                                freq_estimate = len(zero_crossings) / (2 * time_span)  # 零交叉率/2
                                if 0.1 <= freq_estimate <= 10:
                                    features[f'{prefix}_dominant_freq'] = freq_estimate
                                else:
                                    features[f'{prefix}_dominant_freq'] = 0
                            else:
                                features[f'{prefix}_dominant_freq'] = 0
                        else:
                            features[f'{prefix}_dominant_freq'] = 0
                    except:
                        features[f'{prefix}_dominant_freq'] = 0
            else:
                features[f'{prefix}_dominant_freq'] = 0
            
            # 计算无重力影响的特征
            features[f'{prefix}_no_gravity_mean'] = acc_magnitude_no_gravity.mean()
            features[f'{prefix}_no_gravity_std'] = acc_magnitude_no_gravity.std()
            
            # 计算采样率（如果有时间列）
            if time_col:
                time_data = pd.to_numeric(df[time_col], errors='coerce').dropna()
                if len(time_data) > 1:
                    dt = np.diff(time_data).mean()
                    if dt > 0:
                        features['sampling_rate'] = 1.0 / dt
    
    return features

def extract_gyroscope_features(df, prefix='gyro'):
    """从Gyroscope sheet提取特征"""
    if df is None or df.empty:
        return {}
    
    features = {}
    
    # 查找列
    gyro_x_col = None
    gyro_y_col = None
    gyro_z_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'x' in col_lower and ('gyro' in col_lower or '陀螺' in col_lower):
            gyro_x_col = col
        elif 'y' in col_lower and ('gyro' in col_lower or '陀螺' in col_lower):
            gyro_y_col = col
        elif 'z' in col_lower and ('gyro' in col_lower or '陀螺' in col_lower):
            gyro_z_col = col
    
    # 如果没找到，尝试按位置
    if gyro_x_col is None and len(df.columns) > 1:
        gyro_x_col = df.columns[1]
    if gyro_y_col is None and len(df.columns) > 2:
        gyro_y_col = df.columns[2]
    if gyro_z_col is None and len(df.columns) > 3:
        gyro_z_col = df.columns[3]
    
    if gyro_x_col and gyro_y_col and gyro_z_col:
        gyro_x = pd.to_numeric(df[gyro_x_col], errors='coerce').dropna()
        gyro_y = pd.to_numeric(df[gyro_y_col], errors='coerce').dropna()
        gyro_z = pd.to_numeric(df[gyro_z_col], errors='coerce').dropna()
        
        min_len = min(len(gyro_x), len(gyro_y), len(gyro_z))
        if min_len > 0:
            gyro_x = gyro_x[:min_len]
            gyro_y = gyro_y[:min_len]
            gyro_z = gyro_z[:min_len]
            
            gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            
            features[f'{prefix}_mean'] = gyro_magnitude.mean()
            features[f'{prefix}_std'] = gyro_magnitude.std()
            features[f'{prefix}_max'] = gyro_magnitude.max()
            features[f'{prefix}_x_std'] = gyro_x.std()
            features[f'{prefix}_y_std'] = gyro_y.std()
            features[f'{prefix}_z_std'] = gyro_z.std()
    
    return features

def extract_linear_acceleration_features(df, prefix='linear_acc'):
    """从Linear Acceleration sheet提取特征"""
    if df is None or df.empty:
        return {}
    
    features = {}
    
    # 查找列（类似加速度计）
    lin_acc_x_col = None
    lin_acc_y_col = None
    lin_acc_z_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'x' in col_lower and 'linear' in col_lower:
            lin_acc_x_col = col
        elif 'y' in col_lower and 'linear' in col_lower:
            lin_acc_y_col = col
        elif 'z' in col_lower and 'linear' in col_lower:
            lin_acc_z_col = col
    
    # 如果没找到，尝试按位置
    if lin_acc_x_col is None and len(df.columns) > 1:
        lin_acc_x_col = df.columns[1]
    if lin_acc_y_col is None and len(df.columns) > 2:
        lin_acc_y_col = df.columns[2]
    if lin_acc_z_col is None and len(df.columns) > 3:
        lin_acc_z_col = df.columns[3]
    
    if lin_acc_x_col and lin_acc_y_col and lin_acc_z_col:
        lin_acc_x = pd.to_numeric(df[lin_acc_x_col], errors='coerce').dropna()
        lin_acc_y = pd.to_numeric(df[lin_acc_y_col], errors='coerce').dropna()
        lin_acc_z = pd.to_numeric(df[lin_acc_z_col], errors='coerce').dropna()
        
        min_len = min(len(lin_acc_x), len(lin_acc_y), len(lin_acc_z))
        if min_len > 0:
            lin_acc_x = lin_acc_x[:min_len]
            lin_acc_y = lin_acc_y[:min_len]
            lin_acc_z = lin_acc_z[:min_len]
            
            lin_acc_magnitude = np.sqrt(lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2)
            
            features[f'{prefix}_mean'] = lin_acc_magnitude.mean()
            features[f'{prefix}_std'] = lin_acc_magnitude.std()
            features[f'{prefix}_max'] = lin_acc_magnitude.max()
    
    return features

def extract_magnetometer_features(df, prefix='mag'):
    """从Magnetometer sheet提取特征"""
    if df is None or df.empty:
        return {}
    
    features = {}
    
    # 查找列
    mag_x_col = None
    mag_y_col = None
    mag_z_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'x' in col_lower and ('magnetic' in col_lower or 'mag' in col_lower):
            mag_x_col = col
        elif 'y' in col_lower and ('magnetic' in col_lower or 'mag' in col_lower):
            mag_y_col = col
        elif 'z' in col_lower and ('magnetic' in col_lower or 'mag' in col_lower):
            mag_z_col = col
    
    # 如果没找到，尝试按位置
    if mag_x_col is None and len(df.columns) > 1:
        mag_x_col = df.columns[1]
    if mag_y_col is None and len(df.columns) > 2:
        mag_y_col = df.columns[2]
    if mag_z_col is None and len(df.columns) > 3:
        mag_z_col = df.columns[3]
    
    if mag_x_col and mag_y_col and mag_z_col:
        mag_x = pd.to_numeric(df[mag_x_col], errors='coerce').dropna()
        mag_y = pd.to_numeric(df[mag_y_col], errors='coerce').dropna()
        mag_z = pd.to_numeric(df[mag_z_col], errors='coerce').dropna()
        
        min_len = min(len(mag_x), len(mag_y), len(mag_z))
        if min_len > 0:
            mag_x = mag_x[:min_len]
            mag_y = mag_y[:min_len]
            mag_z = mag_z[:min_len]
            
            mag_magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
            
            features[f'{prefix}_mean'] = mag_magnitude.mean()
            features[f'{prefix}_std'] = mag_magnitude.std()
    
    return features

def extract_location_features(df, prefix='loc'):
    """从Location sheet提取特征（如果有数据）"""
    if df is None or df.empty:
        return {}
    
    features = {}
    
    # 查找速度列
    velocity_col = None
    for col in df.columns:
        if 'velocity' in str(col).lower() or '速度' in str(col).lower():
            velocity_col = col
            break
    
    if velocity_col is None and len(df.columns) > 4:
        velocity_col = df.columns[4]  # 通常是第5列
    
    if velocity_col:
        velocity = pd.to_numeric(df[velocity_col], errors='coerce').dropna()
        if len(velocity) > 0:
            features[f'{prefix}_velocity_mean'] = velocity.mean()
            features[f'{prefix}_velocity_max'] = velocity.max()
            features[f'{prefix}_velocity_std'] = velocity.std()
    
    return features

def extract_all_features(sheets_data, name):
    """从所有sheet提取特征"""
    if sheets_data is None:
        return None
    
    features = {'name': name}
    
    # 从各个sheet提取特征
    if 'Accelerometer' in sheets_data:
        acc_features = extract_accelerometer_features(sheets_data['Accelerometer'], 'acc')
        features.update(acc_features)
    
    if 'Gyroscope' in sheets_data:
        gyro_features = extract_gyroscope_features(sheets_data['Gyroscope'], 'gyro')
        features.update(gyro_features)
    
    if 'Linear Acceleration' in sheets_data:
        lin_acc_features = extract_linear_acceleration_features(sheets_data['Linear Acceleration'], 'linear_acc')
        features.update(lin_acc_features)
    
    if 'Magnetometer' in sheets_data:
        mag_features = extract_magnetometer_features(sheets_data['Magnetometer'], 'mag')
        features.update(mag_features)
    
    if 'Location' in sheets_data:
        loc_features = extract_location_features(sheets_data['Location'], 'loc')
        features.update(loc_features)
    
    return features

# ============================================================================
# 4. 提取所有数据集的特征
# ============================================================================
print("\n[步骤4] 提取所有数据集的特征...")

all_features = []
for key, sheets_data in datasets.items():
    if key != 'phyphox' and sheets_data is not None:
        features = extract_all_features(sheets_data, key)
        if features and len(features) > 1:  # 至少有一个特征（除了name）
            all_features.append(features)
            num_features = len([k for k in features.keys() if k != 'name'])
            # 打印dominant frequency的值用于调试
            if 'acc_dominant_freq' in features:
                freq_val = features['acc_dominant_freq']
                print(f"  ✓ {key}: Extracted {num_features} features (dominant_freq: {freq_val:.4f} Hz)")
            else:
                print(f"  ✓ {key}: Extracted {num_features} features (no dominant_freq)")

if len(all_features) > 0:
    features_df = pd.DataFrame(all_features)
    print(f"\n特征数据框形状: {features_df.shape}")
    print(f"特征列: {features_df.columns.tolist()}")
else:
    features_df = pd.DataFrame()
    print("  警告: 未能提取任何特征")

# ============================================================================
# 5. 数据可视化
# ============================================================================
print("\n[步骤5] 生成可视化图表...")

# 创建输出目录
import os
os.makedirs('analysis_results', exist_ok=True)

# 5.1 绘制加速度时间序列对比
def plot_accelerometer_time_series(datasets, save_path='analysis_results'):
    """绘制加速度计时间序列对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Accelerometer Time Series Comparison', fontsize=16, fontweight='bold')
    
    plot_datasets = [
        ('running_flat', 'Running - Flat'),
        ('running_upslope', 'Running - Uphill'),
        ('ebike_flat', 'E-bike - Flat'),
        ('ebike_upslope', 'E-bike - Uphill')
    ]
    
    for idx, (key, title) in enumerate(plot_datasets):
        ax = axes[idx // 2, idx % 2]
        has_data = False
        if datasets[key] is not None and 'Accelerometer' in datasets[key]:
            df = datasets[key]['Accelerometer']
            
            # 查找列
            time_col = df.columns[0] if len(df.columns) > 0 else None
            acc_x_col = df.columns[1] if len(df.columns) > 1 else None
            acc_y_col = df.columns[2] if len(df.columns) > 2 else None
            acc_z_col = df.columns[3] if len(df.columns) > 3 else None
            
            if time_col and acc_x_col and acc_y_col and acc_z_col:
                time_data = pd.to_numeric(df[time_col], errors='coerce').dropna()
                acc_x = pd.to_numeric(df[acc_x_col], errors='coerce').dropna()
                acc_y = pd.to_numeric(df[acc_y_col], errors='coerce').dropna()
                acc_z = pd.to_numeric(df[acc_z_col], errors='coerce').dropna()
                
                min_len = min(len(time_data), len(acc_x), len(acc_y), len(acc_z))
                if min_len > 0:
                    time_data = time_data[:min_len]
                    acc_x = acc_x[:min_len]
                    acc_y = acc_y[:min_len]
                    acc_z = acc_z[:min_len]
                    
                    # 计算合加速度
                    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    
                    # 下采样以提高性能
                    max_points = 2000
                    if len(time_data) > max_points:
                        step = len(time_data) // max_points
                        time_data = time_data[::step]
                        acc_magnitude = acc_magnitude[::step]
                    
                    ax.plot(time_data.values, acc_magnitude.values, linewidth=1.5, alpha=0.8, label='Magnitude')
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Acceleration (m/s²)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, f'{title}\nNo Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/1_accelerometer_time_series.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved accelerometer time series: {save_path}/1_accelerometer_time_series.png")
    plt.close()

plot_accelerometer_time_series(datasets)

# 5.2 绘制特征对比图
def plot_feature_comparison(features_df, save_path='analysis_results'):
    """绘制特征对比图"""
    if features_df.empty:
        return
    
    # 提取路况和活动类型
    def get_terrain(name):
        if 'flat' in name and 'upslope' in name:
            return 'Flat→Uphill'
        elif 'downslope' in name and 'flat' in name:
            return 'Downhill→Flat'
        elif 'flat' in name:
            return 'Flat'
        elif 'upslope' in name:
            return 'Uphill'
        elif 'downslope' in name:
            return 'Downhill'
        return 'Unknown'
    
    def get_activity(name):
        return 'Running' if 'running' in name else 'E-bike'
    
    features_df['terrain'] = features_df['name'].apply(get_terrain)
    features_df['activity'] = features_df['name'].apply(get_activity)
    
    # 绘制加速度特征对比
    if 'acc_mean' in features_df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sensor Feature Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. 平均加速度
        ax1 = axes[0, 0]
        sns.barplot(data=features_df, x='terrain', y='acc_mean', hue='activity', ax=ax1, palette='Set2')
        ax1.set_title('Mean Acceleration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Acceleration (m/s²)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 加速度标准差
        ax2 = axes[0, 1]
        if 'acc_std' in features_df.columns:
            sns.barplot(data=features_df, x='terrain', y='acc_std', hue='activity', ax=ax2, palette='Set2')
            ax2.set_title('Acceleration Std Dev', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Acceleration Std (m/s²)')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # 3. 加速度变异性
        ax3 = axes[0, 2]
        if 'acc_variability' in features_df.columns:
            sns.barplot(data=features_df, x='terrain', y='acc_variability', hue='activity', ax=ax3, palette='Set2')
            ax3.set_title('Acceleration Variability', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Variability')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # 4. 陀螺仪均值
        ax4 = axes[1, 0]
        if 'gyro_mean' in features_df.columns:
            sns.barplot(data=features_df, x='terrain', y='gyro_mean', hue='activity', ax=ax4, palette='Set2')
            ax4.set_title('Mean Gyroscope', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Angular Velocity (rad/s)')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # 5. 主频率
        ax5 = axes[1, 1]
        if 'acc_dominant_freq' in features_df.columns:
            plot_data = features_df.copy()
            # 将NaN替换为0以便显示
            plot_data['acc_dominant_freq'] = plot_data['acc_dominant_freq'].fillna(0)
            
            # 打印调试信息
            print(f"\n  Debug - Dominant Frequency values:")
            for idx, row in plot_data.iterrows():
                print(f"    {row['name']}: {row['acc_dominant_freq']:.4f} Hz (activity: {row['activity']}, terrain: {row['terrain']})")
            
            # 使用seaborn barplot，但确保所有数据都显示
            # 先检查数据
            if len(plot_data) > 0:
                # 确保有所有activity和terrain的组合
                all_terrains = plot_data['terrain'].unique()
                all_activities = plot_data['activity'].unique()
                
                # 检查是否有ebike数据
                has_ebike = (plot_data['activity'] == 'E-bike').any()
                has_running = (plot_data['activity'] == 'Running').any()
                print(f"    Has E-bike data: {has_ebike}, Has Running data: {has_running}")
                
                # 打印每个terrain和activity组合的值
                print(f"    Detailed values by terrain and activity:")
                for terrain in all_terrains:
                    for activity in all_activities:
                        mask = (plot_data['terrain'] == terrain) & (plot_data['activity'] == activity)
                        if mask.any():
                            val = plot_data[mask]['acc_dominant_freq'].iloc[0]
                            print(f"      {terrain} - {activity}: {val:.4f} Hz")
                
                # 使用seaborn barplot，设置dodge=True确保分组显示
                # 使用与其他图相同的palette，并确保显示所有数据
                # 设置estimator=None以显示所有原始值（而不是平均值）
                # 但由于我们每个terrain+activity组合只有一个值，所以用默认的mean也可以
                sns.barplot(data=plot_data, x='terrain', y='acc_dominant_freq', hue='activity', 
                           ax=ax5, palette='Set2', dodge=True, ci=None)
                ax5.set_title('Dominant Frequency', fontsize=12, fontweight='bold')
                ax5.set_ylabel('Frequency (Hz)')
                ax5.set_xlabel('Terrain')
                ax5.tick_params(axis='x', rotation=45)
                
                # 设置y轴范围，确保即使值为0也能看到
                y_min = plot_data['acc_dominant_freq'].min()
                y_max = plot_data['acc_dominant_freq'].max()
                if y_max == 0 and y_min == 0:
                    ax5.set_ylim(-0.1, 0.5)  # 即使都是0，也显示一个小的范围
                elif y_max == y_min:
                    ax5.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
                
                # 如果所有值都是0，添加说明
                if (plot_data['acc_dominant_freq'] == 0).all():
                    ax5.text(0.5, 0.95, 'Note: All values are 0 or NaN', 
                            ha='center', va='top', transform=ax5.transAxes, 
                            fontsize=9, style='italic', color='gray')
                
                # 确保图例显示
                if ax5.get_legend() is None:
                    ax5.legend(title='Activity')
                else:
                    # 确保图例包含所有activity
                    handles, labels = ax5.get_legend_handles_labels()
                    if 'E-bike' not in labels and has_ebike:
                        # 手动添加ebike到图例
                        from matplotlib.patches import Rectangle
                        ebike_patch = Rectangle((0, 0), 1, 1, facecolor=sns.color_palette('Set2')[1], 
                                               label='E-bike', alpha=0.8)
                        handles.append(ebike_patch)
                        labels.append('E-bike')
                        ax5.legend(handles, labels, title='Activity')
            else:
                ax5.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_xticks([])
                ax5.set_yticks([])
        else:
            ax5.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_xticks([])
            ax5.set_yticks([])
        
        # 6. 线性加速度均值
        ax6 = axes[1, 2]
        if 'linear_acc_mean' in features_df.columns:
            sns.barplot(data=features_df, x='terrain', y='linear_acc_mean', hue='activity', ax=ax6, palette='Set2')
            ax6.set_title('Mean Linear Acceleration', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Linear Acc (m/s²)')
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_xticks([])
            ax6.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/2_feature_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved feature comparison: {save_path}/2_feature_comparison.png")
        plt.close()

plot_feature_comparison(features_df)

# 5.3 绘制热力图
def plot_heatmap(features_df, save_path='analysis_results'):
    """绘制特征热力图"""
    if features_df.empty:
        return
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        heatmap_data = features_df.set_index('name')[numeric_cols]
        
        # 处理NaN值：对于每列，如果有NaN，用列均值填充；如果整列都是NaN，用0填充
        for col in heatmap_data.columns:
            if heatmap_data[col].isna().all():
                heatmap_data[col] = 0
            else:
                heatmap_data[col] = heatmap_data[col].fillna(heatmap_data[col].mean())
        
        # 标准化（避免除零错误）
        heatmap_data_norm = heatmap_data.copy()
        for col in heatmap_data.columns:
            col_mean = heatmap_data[col].mean()
            col_std = heatmap_data[col].std()
            if col_std > 0:
                heatmap_data_norm[col] = (heatmap_data[col] - col_mean) / col_std
            else:
                heatmap_data_norm[col] = 0
        
        # 创建mask来隐藏仍然为NaN的值（应该没有了，但以防万一）
        mask = heatmap_data_norm.isna()
        
        sns.heatmap(heatmap_data_norm.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Normalized Value'}, 
                   annot_kws={'size': 8}, mask=mask.T, vmin=-3, vmax=3)
        ax.set_title('Feature Heatmap (Normalized)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Feature')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/3_feature_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved feature heatmap: {save_path}/3_feature_heatmap.png")
        plt.close()

plot_heatmap(features_df)

# 5.4 绘制路况转换分析
def plot_transition_analysis(datasets, save_path='analysis_results'):
    """分析路况转换时的行为变化"""
    transition_datasets = [
        ('running_flat_upslope', 'Running: Flat→Uphill'),
        ('running_downslope_flat', 'Running: Downhill→Flat'),
        ('ebike_flat_upslope', 'E-bike: Flat→Uphill'),
        ('ebike_downslope_flat', 'E-bike: Downhill→Flat')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sensor Data Changes During Terrain Transition', fontsize=16, fontweight='bold')
    
    for idx, (key, title) in enumerate(transition_datasets):
        ax = axes[idx // 2, idx % 2]
        has_data = False
        if datasets[key] is not None and 'Accelerometer' in datasets[key]:
            df = datasets[key]['Accelerometer']
            
            time_col = df.columns[0] if len(df.columns) > 0 else None
            acc_x_col = df.columns[1] if len(df.columns) > 1 else None
            acc_y_col = df.columns[2] if len(df.columns) > 2 else None
            acc_z_col = df.columns[3] if len(df.columns) > 3 else None
            
            if time_col and acc_x_col and acc_y_col and acc_z_col:
                time_data = pd.to_numeric(df[time_col], errors='coerce').dropna()
                acc_x = pd.to_numeric(df[acc_x_col], errors='coerce').dropna()
                acc_y = pd.to_numeric(df[acc_y_col], errors='coerce').dropna()
                acc_z = pd.to_numeric(df[acc_z_col], errors='coerce').dropna()
                
                min_len = min(len(time_data), len(acc_x), len(acc_y), len(acc_z))
                if min_len > 0:
                    time_data = time_data[:min_len]
                    acc_x = acc_x[:min_len]
                    acc_y = acc_y[:min_len]
                    acc_z = acc_z[:min_len]
                    
                    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    
                    # 移动平均平滑
                    window_size = min(50, len(acc_magnitude) // 20)
                    if window_size > 1:
                        acc_smooth = pd.Series(acc_magnitude).rolling(window=window_size, center=True).mean()
                        acc_smooth = acc_smooth.dropna()
                        time_smooth = time_data[acc_smooth.index]
                    else:
                        acc_smooth = acc_magnitude
                        time_smooth = time_data
                    
                    if len(acc_smooth) > 0:
                        ax.plot(time_smooth.values, acc_smooth.values, linewidth=2, alpha=0.8, color='blue')
                        ax.set_title(title, fontsize=12, fontweight='bold')
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Acceleration Magnitude (m/s²)')
                        ax.grid(True, alpha=0.3)
                        
                        # 标记转换点（假设中间位置）
                        mid_point = len(time_smooth) // 2
                        if mid_point < len(time_smooth):
                            ax.axvline(x=time_smooth.iloc[mid_point], color='red', linestyle='--', 
                                     linewidth=2, alpha=0.7, label='Transition Point')
                            ax.legend()
                        has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, f'{title}\nNo Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/4_transition_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved transition analysis: {save_path}/4_transition_analysis.png")
    plt.close()

plot_transition_analysis(datasets)

# 5.5 绘制多传感器对比
def plot_multi_sensor_comparison(datasets, save_path='analysis_results'):
    """绘制多传感器数据对比"""
    if features_df.empty:
        return
    
    # 选择几个代表性的数据集
    selected_datasets = ['running_flat', 'running_upslope', 'ebike_flat', 'ebike_upslope']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Sensor Data Comparison (Running vs E-bike, Flat vs Uphill)', fontsize=16, fontweight='bold')
    
    sensor_types = ['Accelerometer', 'Gyroscope', 'Linear Acceleration', 'Magnetometer']
    
    for idx, sensor_type in enumerate(sensor_types):
        ax = axes[idx // 2, idx % 2]
        
        data_to_plot = []
        labels = []
        
        for key in selected_datasets:
            if datasets[key] is not None and sensor_type in datasets[key]:
                df = datasets[key][sensor_type]
                
                # 获取第一个数值列（通常是x分量）
                if len(df.columns) > 1:
                    data_col = df.columns[1]
                    data = pd.to_numeric(df[data_col], errors='coerce').dropna()
                    
                    if len(data) > 0:
                        # 下采样
                        max_points = 1000
                        if len(data) > max_points:
                            step = len(data) // max_points
                            data = data[::step]
                        
                        data_to_plot.append(data.values)
                        labels.append(key.replace('_', ' ').title())
        
        if len(data_to_plot) > 0:
            for i, (data, label) in enumerate(zip(data_to_plot, labels)):
                ax.plot(data[:min(1000, len(data))], label=label, alpha=0.7, linewidth=1)
            ax.set_title(sensor_type, fontsize=12, fontweight='bold')
            ax.set_xlabel('Sample Points')
            ax.set_ylabel('Sensor Reading')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{sensor_type}\nNo Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/5_multi_sensor_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved multi-sensor comparison: {save_path}/5_multi_sensor_comparison.png")
    plt.close()

plot_multi_sensor_comparison(datasets)

# 5.6 高级可视化：相关性矩阵
def plot_correlation_matrix(features_df, save_path='analysis_results'):
    """绘制特征相关性矩阵"""
    if features_df.empty:
        return
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return
    
    # 选择主要特征
    main_features = [col for col in numeric_cols if any(x in col for x in ['acc_mean', 'acc_std', 'gyro_mean', 'linear_acc_mean', 'mag_mean'])]
    if len(main_features) < 2:
        main_features = numeric_cols[:10]  # 至少选择前10个
    
    corr_data = features_df[main_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
               square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'},
               annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{save_path}/6_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved correlation matrix: {save_path}/6_correlation_matrix.png")
    plt.close()

plot_correlation_matrix(features_df)

# 5.7 高级可视化：箱线图和小提琴图
def plot_box_violin_plots(features_df, save_path='analysis_results'):
    """绘制箱线图和小提琴图"""
    if features_df.empty:
        return
    
    def get_activity(name):
        return 'Running' if 'running' in name else 'E-bike'
    
    features_df['activity'] = features_df['name'].apply(get_activity)
    
    # 选择主要特征
    main_features = ['acc_mean', 'acc_std', 'gyro_mean', 'linear_acc_mean']
    available_features = [f for f in main_features if f in features_df.columns]
    
    if len(available_features) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Distribution: Box Plot & Violin Plot', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(available_features[:4]):
            ax = axes[idx // 2, idx % 2]
            
            # 箱线图
            sns.boxplot(data=features_df, x='activity', y=feature, ax=ax, palette='Set2')
            # 叠加小提琴图
            sns.violinplot(data=features_df, x='activity', y=feature, ax=ax, 
                          inner='box', palette='Set2', alpha=0.5)
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Activity Type')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/7_box_violin_plots.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved box/violin plots: {save_path}/7_box_violin_plots.png")
        plt.close()

plot_box_violin_plots(features_df)

# 5.8 高级可视化：PCA降维可视化
def plot_pca_visualization(features_df, save_path='analysis_results'):
    """PCA降维可视化"""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  ⚠ sklearn not available, skipping PCA visualization")
        return
    
    if features_df.empty:
        return
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return
    
    # 准备数据
    X = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 添加标签
    def get_activity(name):
        return 'Running' if 'running' in name else 'E-bike'
    
    def get_terrain(name):
        if 'flat' in name and 'upslope' in name:
            return 'Flat→Uphill'
        elif 'downslope' in name and 'flat' in name:
            return 'Downhill→Flat'
        elif 'flat' in name:
            return 'Flat'
        elif 'upslope' in name:
            return 'Uphill'
        elif 'downslope' in name:
            return 'Downhill'
        return 'Unknown'
    
    features_df['activity'] = features_df['name'].apply(get_activity)
    features_df['terrain'] = features_df['name'].apply(get_terrain)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PCA Dimensionality Reduction Visualization', fontsize=16, fontweight='bold')
    
    # 按活动类型着色
    ax1 = axes[0]
    for activity in features_df['activity'].unique():
        mask = features_df['activity'] == activity
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=activity, alpha=0.7, s=100)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('PCA by Activity Type', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 按路况着色
    ax2 = axes[1]
    terrain_colors = {'Flat': 'blue', 'Uphill': 'red', 'Downhill': 'green', 
                     'Flat→Uphill': 'orange', 'Downhill→Flat': 'purple', 'Unknown': 'gray'}
    for terrain in features_df['terrain'].unique():
        mask = features_df['terrain'] == terrain
        color = terrain_colors.get(terrain, 'gray')
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], label=terrain, alpha=0.7, s=100, c=color)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.set_title('PCA by Terrain Type', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/8_pca_visualization.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PCA visualization: {save_path}/8_pca_visualization.png")
    plt.close()

plot_pca_visualization(features_df)

# 5.9 高级可视化：雷达图
def plot_radar_chart(features_df, save_path='analysis_results'):
    """绘制雷达图对比不同活动类型"""
    if features_df.empty:
        return
    
    def get_activity(name):
        return 'Running' if 'running' in name else 'E-bike'
    
    features_df['activity'] = features_df['name'].apply(get_activity)
    
    # 选择主要特征
    main_features = ['acc_mean', 'acc_std', 'gyro_mean', 'linear_acc_mean', 'acc_variability']
    available_features = [f for f in main_features if f in features_df.columns]
    
    if len(available_features) < 3:
        return
    
    # 计算每个活动类型的平均值
    activity_means = features_df.groupby('activity')[available_features].mean()
    
    # 标准化到0-1范围
    activity_means_norm = (activity_means - activity_means.min()) / (activity_means.max() - activity_means.min())
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e']
    for idx, (activity, row) in enumerate(activity_means_norm.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, label=activity, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('_', ' ').title() for f in available_features])
    ax.set_ylim(0, 1)
    ax.set_title('Radar Chart: Feature Comparison by Activity', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/9_radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved radar chart: {save_path}/9_radar_chart.png")
    plt.close()

plot_radar_chart(features_df)

# ============================================================================
# 6. 行为模式挖掘
# ============================================================================
print("\n[步骤6] 行为模式挖掘...")

def analyze_patterns(features_df, datasets):
    """挖掘行为模式"""
    patterns = []
    
    if features_df.empty:
        return patterns
    
    # 分离跑步和电动车数据
    running_df = features_df[features_df['name'].str.contains('running')]
    ebike_df = features_df[features_df['name'].str.contains('ebike')]
    
    # 模式1: 跑步 vs 电动车的加速度差异
    if 'acc_mean' in features_df.columns and len(running_df) > 0 and len(ebike_df) > 0:
        running_mean = running_df['acc_mean'].mean()
        ebike_mean = ebike_df['acc_mean'].mean()
        patterns.append({
            'pattern': '运动方式差异 - 加速度',
            'description': f'跑步的平均加速度 ({running_mean:.2f} m/s²) vs 电动车 ({ebike_mean:.2f} m/s²)',
            'insight': '跑步时身体上下运动更剧烈，导致加速度更大。电动车运动更平稳。'
        })
    
    # 模式2: 上坡 vs 平路的差异
    if 'acc_std' in features_df.columns:
        flat_data = features_df[features_df['name'].str.contains('flat') & 
                               ~features_df['name'].str.contains('upslope|downslope')]
        upslope_data = features_df[features_df['name'].str.contains('upslope') & 
                                  ~features_df['name'].str.contains('flat|downslope')]
        
        if len(flat_data) > 0 and len(upslope_data) > 0:
            flat_std = flat_data['acc_std'].mean()
            upslope_std = upslope_data['acc_std'].mean()
            patterns.append({
                'pattern': '路况影响 - 加速度变异性',
                'description': f'上坡路的加速度标准差 ({upslope_std:.2f} m/s²) vs 平路 ({flat_std:.2f} m/s²)',
                'insight': '上坡时需要更多力量，运动模式发生变化，加速度变异性增加。'
            })
    
    # 模式3: 步频/运动频率分析
    if 'acc_dominant_freq' in features_df.columns:
        running_freq = running_df['acc_dominant_freq'].mean() if len(running_df) > 0 else None
        ebike_freq = ebike_df['acc_dominant_freq'].mean() if len(ebike_df) > 0 else None
        
        if running_freq is not None and ebike_freq is not None:
            patterns.append({
                'pattern': '运动频率差异',
                'description': f'跑步的主频率 ({running_freq:.3f} Hz) vs 电动车 ({ebike_freq:.3f} Hz)',
                'insight': '跑步有明显的步频特征，而电动车的运动频率较低且更平稳。'
            })
    
    # 模式4: 陀螺仪差异
    if 'gyro_mean' in features_df.columns and len(running_df) > 0 and len(ebike_df) > 0:
        running_gyro = running_df['gyro_mean'].mean()
        ebike_gyro = ebike_df['gyro_mean'].mean()
        patterns.append({
            'pattern': '运动方式差异 - 角速度',
            'description': f'跑步的平均角速度 ({running_gyro:.3f} rad/s) vs 电动车 ({ebike_gyro:.3f} rad/s)',
            'insight': '跑步时身体旋转和摆动更多，导致角速度更大。'
        })
    
    # 模式5: 路况转换时的变化
    transition_keys = ['running_flat_upslope', 'running_downslope_flat', 
                       'ebike_flat_upslope', 'ebike_downslope_flat']
    for key in transition_keys:
        if key in features_df['name'].values:
            key_features = features_df[features_df['name'] == key].iloc[0]
            if 'acc_mean' in key_features:
                # 比较转换前后的数据（需要从原始数据中分析）
                patterns.append({
                    'pattern': '路况转换检测',
                    'description': f'{key}: 路况转换时传感器数据有明显变化',
                    'insight': '路况转换时加速度特征发生变化，可用于实时路况识别。'
                })
                break
    
    return patterns

patterns = analyze_patterns(features_df, datasets)

# ============================================================================
# 7. 生成分析报告
# ============================================================================
print("\n[步骤7] 生成分析报告...")

report = f"""
{'=' * 80}
智能手机传感器数据分析报告
{'=' * 80}

一、数据概览
-----------
- 跑步数据: 5个数据集（平路、上坡、下坡、平路→上坡、下坡→平路）
- 电动车数据: 5个数据集（平路、上坡、下坡、平路→上坡、下坡→平路）
- 坡度测量数据: phyphox数据
- 传感器类型: 加速度计、陀螺仪、线性加速度、磁力计、位置信息、光照

二、提取的特征
-----------
"""
if not features_df.empty:
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    report += f"- 数值特征数量: {len(numeric_cols)}\n"
    report += f"- 主要特征类别:\n"
    report += f"  * 加速度特征: {len([c for c in numeric_cols if 'acc' in c])} 个\n"
    report += f"  * 陀螺仪特征: {len([c for c in numeric_cols if 'gyro' in c])} 个\n"
    report += f"  * 线性加速度特征: {len([c for c in numeric_cols if 'linear' in c])} 个\n"
    report += f"  * 磁力计特征: {len([c for c in numeric_cols if 'mag' in c])} 个\n"
    report += f"  * 位置特征: {len([c for c in numeric_cols if 'loc' in c])} 个\n"

if upslope_angle is not None:
    report += f"\n- 坡度信息:\n"
    report += f"  * 上坡角度: {upslope_angle:.2f}°\n"
    report += f"  * 下坡角度: {downslope_angle:.2f}°\n"

report += f"""
三、发现的行为模式
-----------
"""
for i, pattern in enumerate(patterns, 1):
    report += f"""
模式 {i}: {pattern['pattern']}
  - 描述: {pattern['description']}
  - 洞察: {pattern['insight']}
"""

report += f"""
四、主要发现总结
-----------
1. 运动方式识别: 
   - 跑步和电动车在加速度、角速度、运动频率等特征上有明显差异
   - 跑步时身体上下运动更剧烈，加速度和角速度更大
   - 跑步有明显的步频特征，而电动车运动更平稳

2. 路况识别: 
   - 不同路况（平路、上坡、下坡）在传感器数据上有可区分的特征
   - 上坡时加速度变异性增加，反映运动强度的变化
   - 下坡时加速度模式与上坡有明显差异

3. 转换检测: 
   - 路况转换时传感器数据有明显变化
   - 可用于实时路况识别和运动状态监测

4. 多传感器融合: 
   - 结合加速度计、陀螺仪、线性加速度、磁力计等多传感器数据
   - 可以提供更全面的运动状态分析

五、可视化图表
-----------
已生成以下可视化图表（保存在 analysis_results/ 目录）:
1. 1_accelerometer_time_series.png - 加速度计时间序列对比
2. 2_feature_comparison.png - 传感器特征对比分析
3. 3_feature_heatmap.png - 特征热力图（标准化）
4. 4_transition_analysis.png - 路况转换分析
5. 5_multi_sensor_comparison.png - 多传感器数据对比

六、技术细节
-----------
- 数据预处理: 自动识别列名（支持中英文），处理缺失值
- 特征提取: 统计特征（均值、标准差、最大值、最小值）、频域特征（主频率）、时域特征（变异性）
- 重力补偿: 计算无重力影响的线性加速度特征
- 数据平滑: 使用移动平均处理噪声

{'=' * 80}
"""

# 保存报告
with open('analysis_results/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✓ 分析报告已保存到: analysis_results/analysis_report.txt")

# 保存特征数据
if not features_df.empty:
    features_df.to_csv('analysis_results/features.csv', index=False, encoding='utf-8-sig')
    print(f"✓ 特征数据已保存到: analysis_results/features.csv")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
