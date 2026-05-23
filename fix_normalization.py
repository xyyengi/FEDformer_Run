"""
修复归一化问题：Solar 使用 MinMaxScaler，Wind/Load 使用 StandardScaler

问题分析：
- StandardScaler 公式: x_scaled = (x - mean) / std
- Solar 原始数据 min=0, mean≈10000+
- 当 Solar=0 时，归一化后变成负值 (0 - mean)/std ≈ -2
- 这导致模型学习到"负值Solar是正常的"，预测时也输出负值

解决方案：
- Solar 使用 MinMaxScaler 归一化到 [0, 1]，保持物理约束
- Wind/Load 使用 StandardScaler，因为它们可以有负值（Wind可能反向）
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

print("="*60)
print("修复归一化问题")
print("="*60)

# 加载原始数据
df = pd.read_csv('D:/FEDformer_Run/Wind_Solar_Load_Processed.csv')
print(f"原始数据形状: {df.shape}")
print(f"原始 Solar 范围: [{df['Solar'].min()}, {df['Solar'].max()}]")
print(f"原始 Solar 均值: {df['Solar'].mean():.2f}")

# 分离数据特征
data_cols = ['Wind', 'Solar', 'Load']
time_cols = ['month', 'day', 'weekday', 'hour']

df_data = df[data_cols].values
df_time = df[time_cols].values

# 数据集划分
num_train = int(len(df) * 0.7)
num_test = int(len(df) * 0.2)
num_vali = len(df) - num_train - num_test

print(f"\n数据集划分:")
print(f"  训练集: 0 ~ {num_train} ({num_train} 样本)")
print(f"  验证集: {num_train} ~ {num_train + num_vali} ({num_vali} 样本)")
print(f"  测试集: {num_train + num_vali} ~ {len(df)} ({num_test} 样本)")

# ============================================================
# 方案1：Solar 使用 MinMaxScaler，Wind/Load 使用 StandardScaler
# ============================================================
print("\n" + "="*60)
print("方案1：Solar MinMaxScaler + Wind/Load StandardScaler")
print("="*60)

# 创建不同的 scaler
wind_scaler = StandardScaler()
solar_scaler = MinMaxScaler(feature_range=(0, 1))  # Solar 归一化到 [0, 1]
load_scaler = StandardScaler()

# 只用训练集拟合
train_data = df_data[:num_train]

wind_scaler.fit(train_data[:, 0:1])   # Wind
solar_scaler.fit(train_data[:, 1:2])  # Solar
load_scaler.fit(train_data[:, 2:3])   # Load

print(f"\nScaler 参数:")
print(f"  Wind: mean={wind_scaler.mean_[0]:.2f}, std={np.sqrt(wind_scaler.var_[0]):.2f}")
print(f"  Solar: min={solar_scaler.data_min_[0]:.2f}, max={solar_scaler.data_max_[0]:.2f}")
print(f"  Load: mean={load_scaler.mean_[0]:.2f}, std={np.sqrt(load_scaler.var_[0]):.2f}")

# 分别归一化
wind_scaled = wind_scaler.transform(df_data[:, 0:1])
solar_scaled = solar_scaler.transform(df_data[:, 1:2])
load_scaled = load_scaler.transform(df_data[:, 2:3])

# 合并
data_scaled = np.concatenate([wind_scaled, solar_scaled, load_scaled], axis=1)

print(f"\n归一化后数据范围:")
print(f"  Wind: [{wind_scaled.min():.2f}, {wind_scaled.max():.2f}]")
print(f"  Solar: [{solar_scaled.min():.2f}, {solar_scaled.max():.2f}]  <-- 注意：最小值是 0，不是负数！")
print(f"  Load: [{load_scaled.min():.2f}, {load_scaled.max():.2f}]")

# 验证反归一化
wind_inv = wind_scaler.inverse_transform(wind_scaled)
solar_inv = solar_scaler.inverse_transform(solar_scaled)
load_inv = load_scaler.inverse_transform(load_scaled)

print(f"\n反归一化验证:")
print(f"  Solar 反归一化后范围: [{solar_inv.min():.2f}, {solar_inv.max():.2f}]")
print(f"  Solar 负值数量: {(solar_inv < 0).sum()}  <-- 应该是 0！")

# 保存 scaler 参数
output_dir = 'D:/FEDformer_Run/results/4.27/scalers'
os.makedirs(output_dir, exist_ok=True)

scaler_params = {
    'wind': {'mean': wind_scaler.mean_[0], 'var': wind_scaler.var_[0]},
    'solar': {'min': solar_scaler.data_min_[0], 'max': solar_scaler.data_max_[0], 'scale': solar_scaler.scale_[0]},
    'load': {'mean': load_scaler.mean_[0], 'var': load_scaler.var_[0]}
}

np.savez(os.path.join(output_dir, 'scaler_params.npz'), **scaler_params)
print(f"\nScaler 参数已保存到: {output_dir}/scaler_params.npz")

# ============================================================
# 方案2：全部使用 MinMaxScaler（更简单）
# ============================================================
print("\n" + "="*60)
print("方案2：全部使用 MinMaxScaler")
print("="*60)

minmax_scaler = MinMaxScaler(feature_range=(0, 1))
minmax_scaler.fit(train_data)
data_minmax = minmax_scaler.transform(df_data)

print(f"MinMaxScaler 归一化后范围:")
print(f"  Wind: [{data_minmax[:, 0].min():.2f}, {data_minmax[:, 0].max():.2f}]")
print(f"  Solar: [{data_minmax[:, 1].min():.2f}, {data_minmax[:, 1].max():.2f}]")
print(f"  Load: [{data_minmax[:, 2].min():.2f}, {data_minmax[:, 2].max():.2f}]")

# 验证反归一化
data_inv = minmax_scaler.inverse_transform(data_minmax)
print(f"\n反归一化验证:")
print(f"  Solar 反归一化后范围: [{data_inv[:, 1].min():.2f}, {data_inv[:, 1].max():.2f}]")
print(f"  Solar 负值数量: {(data_inv[:, 1] < 0).sum()}")

# ============================================================
# 推荐方案
# ============================================================
print("\n" + "="*60)
print("推荐方案")
print("="*60)

print("""
【推荐使用方案1】
- Solar 使用 MinMaxScaler 归一化到 [0, 1]
- Wind/Load 使用 StandardScaler

优点：
1. Solar 保持物理约束，归一化后最小值为 0
2. Wind/Load 保持 StandardScaler 的统计特性
3. 反归一化后 Solar 不会有负值

【修改 data_loader.py 的方法】
1. 导入 MinMaxScaler: from sklearn.preprocessing import StandardScaler, MinMaxScaler
2. 创建 solar_scaler = MinMaxScaler(feature_range=(0, 1))
3. 分别对 Wind, Solar, Load 进行归一化
4. 在 inverse_transform 中分别反归一化

【或者更简单：全部使用 MinMaxScaler】
- 所有特征归一化到 [0, 1]
- 反归一化后都不会有负值
- 但可能损失一些统计特性
""")