"""
测试 scaler 逻辑（不需要 torch）
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("测试 Scaler 逻辑")
print("=" * 60)

# 加载原始数据
df = pd.read_csv('Wind_Solar_Load_Processed.csv')
print(f"\n原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 提取数据特征
data_cols = ['Wind', 'Solar', 'Load']
data = df[data_cols].values

print(f"\n数据特征形状: {data.shape}")
print(f"Wind 范围: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]")
print(f"Solar 范围: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]")
print(f"Load 范围: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]")

# 训练集划分
num_train = int(len(df) * 0.7)
train_data = data[:num_train]

print(f"\n训练集大小: {num_train}")

# ============ 测试 MinMaxScaler ============
print("\n" + "=" * 60)
print("测试 MinMaxScaler")
print("=" * 60)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)

print(f"\nScaler 参数:")
print(f"  data_min_: {scaler.data_min_}")
print(f"  data_max_: {scaler.data_max_}")
print(f"  scale_: {scaler.scale_}")
print(f"  min_: {scaler.min_}")

# 归一化
normalized = scaler.transform(data)
print(f"\n归一化后范围:")
print(f"  Wind: [{normalized[:, 0].min():.4f}, {normalized[:, 0].max():.4f}]")
print(f"  Solar: [{normalized[:, 1].min():.4f}, {normalized[:, 1].max():.4f}]")
print(f"  Load: [{normalized[:, 2].min():.4f}, {normalized[:, 2].max():.4f}]")

# 反归一化
denormalized = scaler.inverse_transform(normalized)
print(f"\n反归一化后范围:")
print(f"  Wind: [{denormalized[:, 0].min():.2f}, {denormalized[:, 0].max():.2f}]")
print(f"  Solar: [{denormalized[:, 1].min():.2f}, {denormalized[:, 1].max():.2f}]")
print(f"  Load: [{denormalized[:, 2].min():.2f}, {denormalized[:, 2].max():.2f}]")

# 检查是否与原始数据一致
diff = np.abs(denormalized - data)
print(f"\n反归一化误差 (应该为0):")
print(f"  Wind max diff: {diff[:, 0].max():.10f}")
print(f"  Solar max diff: {diff[:, 1].max():.10f}")
print(f"  Load max diff: {diff[:, 2].max():.10f}")

# ============ 测试负值场景 ============
print("\n" + "=" * 60)
print("测试负值场景")
print("=" * 60)

# 模拟模型输出负值
test_pred_normalized = np.array([
    [-0.5, -0.3, 0.5],  # Wind 和 Solar 都是负值
    [0.0, 0.0, 0.5],    # 边界值
    [0.5, 0.5, 0.5],    # 正常值
    [1.5, 1.5, 1.5],    # 超出范围
])

print(f"\n模拟归一化后的预测值:")
print(test_pred_normalized)

# 反归一化
test_pred_denorm = scaler.inverse_transform(test_pred_normalized)
print(f"\n反归一化后的预测值:")
print(test_pred_denorm)

print(f"\n结论:")
print(f"  Wind 反归一化后: [{test_pred_denorm[:, 0].min():.2f}, {test_pred_denorm[:, 0].max():.2f}]")
print(f"  Solar 反归一化后: [{test_pred_denorm[:, 1].min():.2f}, {test_pred_denorm[:, 1].max():.2f}]")

# 检查负值
wind_neg = np.sum(test_pred_denorm[:, 0] < 0)
solar_neg = np.sum(test_pred_denorm[:, 1] < 0)
print(f"\n负值数量:")
print(f"  Wind: {wind_neg}")
print(f"  Solar: {solar_neg}")

if solar_neg > 0:
    print("\n⚠️ MinMaxScaler 反归一化后 Solar 仍可能有负值！")
    print("   原因：模型在归一化空间输出负值，反归一化后仍为负值")
    print("   解决方案：需要在反归一化后对 Solar 做 clip(0, max)")
else:
    print("\n✅ MinMaxScaler 反归一化后 Solar 没有负值")

# ============ 测试 clip 方案 ============
print("\n" + "=" * 60)
print("测试 clip 方案")
print("=" * 60)

test_pred_clipped = test_pred_denorm.copy()
test_pred_clipped[:, 1] = np.clip(test_pred_clipped[:, 1], 0, None)  # Solar clip 到 >= 0

print(f"\nclip 后的 Solar: {test_pred_clipped[:, 1]}")
print(f"负值数量: {np.sum(test_pred_clipped[:, 1] < 0)}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
关键发现：
1. MinMaxScaler 归一化范围是 [0, 1]
2. 如果模型在归一化空间输出负值，反归一化后仍为负值
3. Solar 的 data_min_ = 0，所以：
   - 归一化值 = (原始值 - 0) / max
   - 反归一化值 = 归一化值 * max
   - 如果归一化值 < 0，反归一化值 < 0

解决方案：
1. 在反归一化后对 Solar 做 clip(0, max)
2. 或者在模型输出层加 ReLU/Sigmoid 约束
""")