import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载测试集结果
results_dir = 'D:/FEDformer_Run/results/4.27'
test_pred = np.load(os.path.join(results_dir, 'test_pred.npy'))
test_res = np.load(os.path.join(results_dir, 'test_res.npy'))

print(f"test_pred shape: {test_pred.shape}")
print(f"test_res shape: {test_res.shape}")

# 特征名称
feature_names = ['Wind', 'Solar', 'Load']

# 创建输出目录
output_dir = 'D:/FEDformer_Run/results/4.27/figures'
os.makedirs(output_dir, exist_ok=True)

# 1. 绘制每个特征的预测对比曲线（取前500个样本的展开视图）
n_samples_to_show = 200  # 展示的样本数量

for i, name in enumerate(feature_names):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：预测 vs 真实（展开为连续时间序列）
    pred_flat = test_pred[:n_samples_to_show, :, i].flatten()
    res_flat = test_res[:n_samples_to_show, :, i].flatten()
    true_flat = pred_flat + res_flat  # 真实值 = 预测值 + 残差
    
    axes[0].plot(true_flat, label='真实值', color='blue', alpha=0.7, linewidth=1)
    axes[0].plot(pred_flat, label='预测值', color='red', alpha=0.7, linewidth=1)
    axes[0].set_title(f'{name} - 测试集预测对比（前{n_samples_to_show}个样本展开）', fontsize=14)
    axes[0].set_xlabel('时间步（展开）', fontsize=12)
    axes[0].set_ylabel('值', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 下图：残差分布
    axes[1].plot(res_flat, label='残差', color='green', alpha=0.7, linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title(f'{name} - 残差分析', fontsize=14)
    axes[1].set_xlabel('时间步（展开）', fontsize=12)
    axes[1].set_ylabel('残差 (Actual - Forecast)', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_test_comparison.png'), dpi=150)
    plt.close()
    print(f"已保存: {name}_test_comparison.png")

# 2. 绘制单样本详细对比（取第0个样本）
sample_idx = 0
pred_len = test_pred.shape[1]  # 预测长度

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for i, name in enumerate(feature_names):
    pred_sample = test_pred[sample_idx, :, i]
    res_sample = test_res[sample_idx, :, i]
    true_sample = pred_sample + res_sample
    
    axes[i].plot(range(pred_len), true_sample, label='真实值', color='blue', marker='o', markersize=3, alpha=0.7)
    axes[i].plot(range(pred_len), pred_sample, label='预测值', color='red', marker='x', markersize=3, alpha=0.7)
    axes[i].fill_between(range(pred_len), pred_sample, true_sample, alpha=0.2, color='gray', label='误差区间')
    axes[i].set_title(f'{name} - 样本{sample_idx}的168步预测', fontsize=14)
    axes[i].set_xlabel('预测步数', fontsize=12)
    axes[i].set_ylabel('值', fontsize=12)
    axes[i].legend(fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'single_sample_168step.png'), dpi=150)
plt.close()
print(f"已保存: single_sample_168step.png")

# 3. 绘制统计指标
print("\n===== 测试集统计指标 =====")
for i, name in enumerate(feature_names):
    pred_all = test_pred[:, :, i].flatten()
    res_all = test_res[:, :, i].flatten()
    true_all = pred_all + res_all
    
    mse = np.mean(res_all ** 2)
    mae = np.mean(np.abs(res_all))
    rmse = np.sqrt(mse)
    
    # 计算峰值误差
    peak_true = np.max(true_all)
    peak_pred = np.max(pred_all)
    peak_error = peak_pred - peak_true
    
    # 计算谷值误差
    valley_true = np.min(true_all)
    valley_pred = np.min(pred_all)
    valley_error = valley_pred - valley_true
    
    print(f"\n{name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  真实峰值: {peak_true:.2f}, 预测峰值: {peak_pred:.2f}, 峰值误差: {peak_error:.2f}")
    print(f"  真实谷值: {valley_true:.2f}, 预测谷值: {valley_pred:.2f}, 谷值误差: {valley_error:.2f}")
    
    # Solar 特殊检查：负值数量
    if name == 'Solar':
        negative_count = np.sum(pred_all < 0)
        negative_ratio = negative_count / len(pred_all) * 100
        print(f"  Solar负值数量: {negative_count} ({negative_ratio:.2f}%)")

# 4. 绘制整体趋势对比（所有测试数据）
fig, axes = plt.subplots(3, 1, figsize=(20, 12))

for i, name in enumerate(feature_names):
    pred_all = test_pred[:, :, i].flatten()
    res_all = test_res[:, :, i].flatten()
    true_all = pred_all + res_all
    
    # 只展示部分数据避免图太密集
    show_len = min(5000, len(true_all))
    
    axes[i].plot(true_all[:show_len], label='真实值', color='blue', alpha=0.5, linewidth=0.8)
    axes[i].plot(pred_all[:show_len], label='预测值', color='red', alpha=0.5, linewidth=0.8)
    axes[i].set_title(f'{name} - 测试集整体趋势对比', fontsize=14)
    axes[i].set_xlabel('时间步（展开）', fontsize=12)
    axes[i].set_ylabel('值', fontsize=12)
    axes[i].legend(fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overall_trend.png'), dpi=150)
plt.close()
print(f"已保存: overall_trend.png")

print(f"\n所有图片已保存到: {output_dir}")
