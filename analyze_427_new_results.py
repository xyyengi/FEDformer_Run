"""
分析 4.27 重新预测后的结果
重点检查 Solar clip 是否生效（负值是否为 0）
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载结果
result_dir = './results/4.27'
pred = np.load(os.path.join(result_dir, 'pred.npy'))
true = np.load(os.path.join(result_dir, 'true.npy'))

print("=" * 60)
print("4.27 重新预测结果分析 (Solar clip 已应用)")
print("=" * 60)

# 基本信息
print(f"\n【数据形状】")
print(f"pred.shape: {pred.shape}")
print(f"true.shape: {true.shape}")

# 特征名称
feature_names = ['Wind', 'Solar', 'Load']

# 各特征统计
print(f"\n【各特征统计】")
for i, name in enumerate(feature_names):
    pred_feat = pred[:, :, i]
    true_feat = true[:, :, i]
    
    # 负值检查
    neg_count = np.sum(pred_feat < 0)
    neg_ratio = neg_count / pred_feat.size * 100
    min_val = pred_feat.min()
    
    # MAE 和相关系数
    mae = np.mean(np.abs(pred_feat - true_feat))
    corr = np.corrcoef(pred_feat.flatten(), true_feat.flatten())[0, 1]
    
    print(f"\n{name}:")
    print(f"  预测范围: [{pred_feat.min():.2f}, {pred_feat.max():.2f}]")
    print(f"  真实范围: [{true_feat.min():.2f}, {true_feat.max():.2f}]")
    print(f"  负值数量: {neg_count} ({neg_ratio:.2f}%)")
    print(f"  最小值: {min_val:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  相关系数: {corr:.4f}")

# 重点检查 Solar
print("\n" + "=" * 60)
print("【Solar 负值检查 - 重点】")
print("=" * 60)
solar_pred = pred[:, :, 1]
solar_true = true[:, :, 1]

neg_count = np.sum(solar_pred < 0)
print(f"Solar 负值数量: {neg_count}")
print(f"Solar 负值比例: {neg_count / solar_pred.size * 100:.4f}%")
print(f"Solar 预测最小值: {solar_pred.min():.4f}")

if neg_count == 0:
    print("[OK] Solar clip 已生效！所有 Solar 预测值 >= 0")
else:
    print(f"[WARN] Solar clip 未完全生效，仍有 {neg_count} 个负值")
    # 显示负值样本
    neg_indices = np.where(solar_pred < 0)
    print(f"负值样本位置 (前10个):")
    for j in range(min(10, len(neg_indices[0]))):
        sample_idx = neg_indices[0][j]
        step_idx = neg_indices[1][j]
        print(f"  样本 {sample_idx}, 步长 {step_idx}: pred={solar_pred[sample_idx, step_idx]:.2f}, true={solar_true[sample_idx, step_idx]:.2f}")

# 生成可视化
output_dir = os.path.join(result_dir, 'analysis_new')
os.makedirs(output_dir, exist_ok=True)

# 1. 各特征预测 vs 真实对比图
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, name in enumerate(feature_names):
    ax = axes[i]
    # 取第一个样本
    pred_sample = pred[0, :, i]
    true_sample = true[0, :, i]
    
    ax.plot(pred_sample, label='Pred', alpha=0.7)
    ax.plot(true_sample, label='True', alpha=0.7)
    ax.set_title(f'{name} - Sample 0: Pred vs True')
    ax.set_xlabel('Time Step')
    ax.set_ylabel(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_comparison.png'), dpi=150)
print(f"\n可视化已保存: {output_dir}/feature_comparison.png")

# 2. Solar 详细分析图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1 Solar 预测分布
ax = axes[0, 0]
ax.hist(solar_pred.flatten(), bins=50, alpha=0.7, label='Pred')
ax.hist(solar_true.flatten(), bins=50, alpha=0.7, label='True')
ax.set_title('Solar Value Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Count')
ax.legend()

# 2.2 Solar 散点图
ax = axes[0, 1]
ax.scatter(solar_true.flatten(), solar_pred.flatten(), alpha=0.3, s=1)
ax.plot([0, solar_true.max()], [0, solar_true.max()], 'r--', label='Ideal')
ax.set_title('Solar: Pred vs True Scatter')
ax.set_xlabel('True')
ax.set_ylabel('Pred')
ax.legend()

# 2.3 Solar 误差分布
ax = axes[1, 0]
error = solar_true - solar_pred
ax.hist(error.flatten(), bins=50, alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--')
ax.set_title(f'Solar Residual Distribution (MAE={np.mean(np.abs(error)):.2f})')
ax.set_xlabel('Residual (Actual - Forecast)')
ax.set_ylabel('Count')

# 2.4 Solar 时间序列对比 (多个样本)
ax = axes[1, 1]
for sample_idx in [0, 5, 10]:
    ax.plot(solar_pred[sample_idx, :], label=f'Pred sample {sample_idx}', alpha=0.7)
    ax.plot(solar_true[sample_idx, :], label=f'True sample {sample_idx}', alpha=0.5, linestyle='--')
ax.set_title('Solar: Multiple Samples Comparison')
ax.set_xlabel('Time Step')
ax.set_ylabel('Solar')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'solar_analysis.png'), dpi=150)
print(f"可视化已保存: {output_dir}/solar_analysis.png")

# 3. 夜间 Solar 检查 (假设夜间是某些时段)
# 这里我们检查 Solar 真实值为 0 时，预测值是否也为 0
print("\n" + "=" * 60)
print("【夜间 Solar 检查】")
print("=" * 60)
night_mask = solar_true == 0
night_pred = solar_pred[night_mask]
night_pred_neg = np.sum(night_pred < 0)
night_pred_positive = np.sum(night_pred > 0)

print(f"夜间样本数量 (True=0): {night_mask.sum()}")
print(f"夜间预测负值数量: {night_pred_neg}")
print(f"夜间预测正值数量: {night_pred_positive}")
print(f"夜间预测均值: {night_pred.mean():.4f}")
print(f"夜间预测最大值: {night_pred.max():.4f}")

if night_pred_neg == 0:
    print("[OK] 夜间 Solar 无负值")
else:
    print(f"[WARN] 夜间 Solar 有 {night_pred_neg} 个负值")

# 保存分析报告
report_path = os.path.join(output_dir, 'analysis_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("4.27 重新预测结果分析报告\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("【Solar clip 效果检查】\n")
    f.write(f"Solar 负值数量: {neg_count}\n")
    f.write(f"Solar 负值比例: {neg_count / solar_pred.size * 100:.4f}%\n")
    f.write(f"Solar 预测最小值: {solar_pred.min():.4f}\n")
    
    if neg_count == 0:
        f.write("✅ Solar clip 已生效！\n\n")
    else:
        f.write(f"❌ Solar clip 未完全生效\n\n")
    
    f.write("【各特征指标】\n")
    for i, name in enumerate(feature_names):
        pred_feat = pred[:, :, i]
        true_feat = true[:, :, i]
        mae = np.mean(np.abs(pred_feat - true_feat))
        corr = np.corrcoef(pred_feat.flatten(), true_feat.flatten())[0, 1]
        f.write(f"{name}: MAE={mae:.2f}, Corr={corr:.4f}\n")
    
    f.write("\n【夜间 Solar 检查】\n")
    f.write(f"夜间样本数量: {night_mask.sum()}\n")
    f.write(f"夜间预测负值: {night_pred_neg}\n")
    f.write(f"夜间预测正值: {night_pred_positive}\n")

print(f"\n分析报告已保存: {report_path}")
print("\n分析完成！")
