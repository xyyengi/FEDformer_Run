"""
分析 4.27 训练结果
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
pred = np.load('results/4.27/pred.npy')
true = np.load('results/4.27/true.npy')
train_pred = np.load('results/4.27/train_pred.npy')
val_pred = np.load('results/4.27/val_pred.npy')
test_pred = np.load('results/4.27/test_pred.npy')

feature_names = ['Wind', 'Solar', 'Load']

print("=" * 60)
print("4.27 训练结果分析报告")
print("=" * 60)

print(f"\n数据形状:")
print(f"  pred shape: {pred.shape}")
print(f"  train_pred shape: {train_pred.shape}")
print(f"  val_pred shape: {val_pred.shape}")
print(f"  test_pred shape: {test_pred.shape}")

# ============ 各特征详细分析 ============
print("\n" + "=" * 60)
print("各特征详细分析")
print("=" * 60)

for i, name in enumerate(feature_names):
    pred_i = pred[:, :, i]
    true_i = true[:, :, i]
    
    mae = np.mean(np.abs(pred_i - true_i))
    mse = np.mean((pred_i - true_i) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_i - pred_i) / (true_i + 1e-8))) * 100
    
    neg_count = np.sum(pred_i < 0)
    neg_pct = neg_count / pred_i.size * 100
    
    print(f"\n【{name}】")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  预测值范围: [{pred_i.min():.2f}, {pred_i.max():.2f}]")
    print(f"  真实值范围: [{true_i.min():.2f}, {true_i.max():.2f}]")
    print(f"  负值数量: {neg_count} / {pred_i.size} ({neg_pct:.2f}%)")

# ============ 滞后分析 ============
print("\n" + "=" * 60)
print("滞后分析 (计算预测与真实值的相关性)")
print("=" * 60)

for i, name in enumerate(feature_names):
    pred_i = pred[:, :, i].flatten()
    true_i = true[:, :, i].flatten()
    
    # 计算不同滞后步数的相关系数
    correlations = []
    for lag in range(0, 25):
        if lag == 0:
            corr = np.corrcoef(pred_i, true_i)[0, 1]
        else:
            corr = np.corrcoef(pred_i[lag:], true_i[:-lag])[0, 1]
        correlations.append(corr)
    
    best_lag = np.argmax(correlations)
    print(f"\n【{name}】")
    print(f"  当前步相关系数: {correlations[0]:.4f}")
    print(f"  最佳滞后步数: {best_lag} (相关系数: {correlations[best_lag]:.4f})")
    if best_lag > 0:
        print(f"  ⚠️ 存在 {best_lag} 步滞后!")

# ============ 生成可视化图表 ============
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 图1: 各特征预测对比 (取最后一个样本的168小时)
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

for i, (ax, name) in enumerate(zip(axes, feature_names)):
    # 取最后一个样本
    pred_sample = pred[-1, :, i]
    true_sample = true[-1, :, i]
    
    hours = np.arange(168)
    ax.plot(hours, true_sample, 'b-', label='True', linewidth=1.5, alpha=0.8)
    ax.plot(hours, pred_sample, 'r-', label='Pred', linewidth=1.5, alpha=0.8)
    ax.fill_between(hours, 0, pred_sample, where=(pred_sample < 0), 
                    color='red', alpha=0.3, label='Negative Region')
    ax.set_title(f'{name} - Last Sample (168h)', fontsize=12)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/4.27/analysis/feature_comparison.png', dpi=150)
plt.close()
print("  保存: feature_comparison.png")

# 图2: Solar 负值分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

solar_pred = pred[:, :, 1].flatten()
solar_true = true[:, :, 1].flatten()

# 负值分布
axes[0].hist(solar_pred[solar_pred < 0], bins=50, color='red', alpha=0.7)
axes[0].set_title('Solar Negative Predictions Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count')

# 预测vs真实散点图
axes[1].scatter(solar_true, solar_pred, alpha=0.1, s=1)
axes[1].plot([solar_true.min(), solar_true.max()], 
             [solar_true.min(), solar_true.max()], 'r--', label='Perfect')
axes[1].axhline(y=0, color='g', linestyle='--', label='Zero Line')
axes[1].set_title('Solar: Predicted vs True')
axes[1].set_xlabel('True')
axes[1].set_ylabel('Predicted')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/4.27/analysis/solar_analysis.png', dpi=150)
plt.close()
print("  保存: solar_analysis.png")

# 图3: Wind 分析
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wind_pred = pred[:, :, 0].flatten()
wind_true = true[:, :, 0].flatten()

# 预测vs真实散点图
axes[0].scatter(wind_true, wind_pred, alpha=0.1, s=1)
axes[0].plot([wind_true.min(), wind_true.max()], 
             [wind_true.min(), wind_true.max()], 'r--', label='Perfect')
axes[0].set_title('Wind: Predicted vs True')
axes[0].set_xlabel('True')
axes[0].set_ylabel('Predicted')
axes[0].legend()

# 残差分布
residuals = wind_true - wind_pred
axes[1].hist(residuals, bins=50, color='blue', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_title(f'Wind Residuals (Mean: {residuals.mean():.2f})')
axes[1].set_xlabel('Residual (Actual - Forecast)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('results/4.27/analysis/wind_analysis.png', dpi=150)
plt.close()
print("  保存: wind_analysis.png")

# 图4: Load 分析
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

load_pred = pred[:, :, 2].flatten()
load_true = true[:, :, 2].flatten()

# 预测vs真实散点图
axes[0].scatter(load_true, load_pred, alpha=0.1, s=1)
axes[0].plot([load_true.min(), load_true.max()], 
             [load_true.min(), load_true.max()], 'r--', label='Perfect')
axes[0].set_title('Load: Predicted vs True')
axes[0].set_xlabel('True')
axes[0].set_ylabel('Predicted')
axes[0].legend()

# 残差分布
residuals = load_true - load_pred
axes[1].hist(residuals, bins=50, color='green', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].set_title(f'Load Residuals (Mean: {residuals.mean():.2f})')
axes[1].set_xlabel('Residual (Actual - Forecast)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('results/4.27/analysis/load_analysis.png', dpi=150)
plt.close()
print("  保存: load_analysis.png")

# 图5: 多样本对比 (取3个不同时间段的样本)
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

sample_indices = [0, len(pred)//2, len(pred)-1]  # 开始、中间、结束

for row, idx in enumerate(sample_indices):
    for col, (i, name) in enumerate(zip(range(3), feature_names)):
        pred_sample = pred[idx, :, i]
        true_sample = true[idx, :, i]
        
        hours = np.arange(168)
        axes[row, col].plot(hours, true_sample, 'b-', label='True', linewidth=1.5)
        axes[row, col].plot(hours, pred_sample, 'r-', label='Pred', linewidth=1.5)
        axes[row, col].set_title(f'{name} - Sample {idx}')
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/4.27/analysis/multi_sample_comparison.png', dpi=150)
plt.close()
print("  保存: multi_sample_comparison.png")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
