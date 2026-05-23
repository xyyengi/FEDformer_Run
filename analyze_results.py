import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

# 计算真实值
test_true = test_pred + test_res

# 创建输出目录
output_dir = 'D:/FEDformer_Run/results/4.27/analysis'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. 最后一段168小时的预测曲线
# ============================================================
print("\n" + "="*60)
print("1. 最后一段168小时预测曲线")
print("="*60)

# 取最后一个样本（包含168小时预测）
last_sample_idx = -1
pred_168 = test_pred[last_sample_idx, :, :]  # shape: (168, 11)
true_168 = test_true[last_sample_idx, :, :]  # shape: (168, 11)
res_168 = test_res[last_sample_idx, :, :]    # shape: (168, 11)

print(f"最后168小时数据形状: {pred_168.shape}")

# 绘制风光负荷三者的168小时预测对比
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for i, name in enumerate(feature_names):
    hours = np.arange(1, 169)  # 1-168小时
    
    axes[i].plot(hours, true_168[:, i], label='真实值', color='blue', linewidth=2, marker='o', markersize=3)
    axes[i].plot(hours, pred_168[:, i], label='预测值', color='red', linewidth=2, marker='x', markersize=3)
    axes[i].fill_between(hours, pred_168[:, i], true_168[:, i], alpha=0.3, color='gray', label='误差区间')
    
    axes[i].set_title(f'{name} - 最后168小时预测对比', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('小时 (h)', fontsize=12)
    axes[i].set_ylabel('出力 (MW)', fontsize=12)
    axes[i].legend(fontsize=11, loc='upper right')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim(1, 168)
    
    # 添加统计信息
    mae = np.mean(np.abs(res_168[:, i]))
    rmse = np.sqrt(np.mean(res_168[:, i]**2))
    axes[i].text(0.02, 0.95, f'MAE: {mae:.2f}  RMSE: {rmse:.2f}', 
                 transform=axes[i].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'last_168h_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: last_168h_comparison.png")

# ============================================================
# 2. 残差分布分析
# ============================================================
print("\n" + "="*60)
print("2. 残差分布分析")
print("="*60)

# 展平所有残差
all_res = {
    'Wind': test_res[:, :, 0].flatten(),
    'Solar': test_res[:, :, 1].flatten(),
    'Load': test_res[:, :, 2].flatten()
}

# 绘制残差分布直方图
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for i, name in enumerate(feature_names):
    res_flat = all_res[name]
    
    # 左图：残差直方图
    axes[i, 0].hist(res_flat, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 拟合正态分布
    mu, std = stats.norm.fit(res_flat)
    x = np.linspace(res_flat.min(), res_flat.max(), 100)
    axes[i, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'正态拟合\nμ={mu:.2f}, σ={std:.2f}')
    
    axes[i, 0].set_title(f'{name} 残差分布直方图', fontsize=14, fontweight='bold')
    axes[i, 0].set_xlabel('Residual (Actual - Forecast)', fontsize=12)
    axes[i, 0].set_ylabel('密度', fontsize=12)
    axes[i, 0].legend(fontsize=10)
    axes[i, 0].grid(True, alpha=0.3)
    
    # 右图：残差QQ图
    stats.probplot(res_flat, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(f'{name} 残差 Q-Q 图', fontsize=14, fontweight='bold')
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: residual_distribution.png")

# 残差统计指标
print("\n残差统计指标:")
print("-" * 80)
print(f"{'特征':<10} {'均值':<15} {'标准差':<15} {'偏度':<15} {'峰度':<15} {'正态性(p)':<15}")
print("-" * 80)

residual_stats = {}
for name in feature_names:
    res_flat = all_res[name]
    
    mean = np.mean(res_flat)
    std = np.std(res_flat)
    skewness = stats.skew(res_flat)
    kurtosis = stats.kurtosis(res_flat)
    
    # Shapiro-Wilk 正态性检验（取子样本，因为数据量太大）
    sample_size = min(5000, len(res_flat))
    sample_idx = np.random.choice(len(res_flat), sample_size, replace=False)
    stat, p_value = stats.shapiro(res_flat[sample_idx])
    
    residual_stats[name] = {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'normality_p': p_value
    }
    
    print(f"{name:<10} {mean:<15.2f} {std:<15.2f} {skewness:<15.4f} {kurtosis:<15.4f} {p_value:<15.6f}")

print("-" * 80)

# 残差相关性分析
print("\n残差相关性矩阵:")
residual_corr = np.corrcoef([all_res['Wind'], all_res['Solar'], all_res['Load']])
print(f"{'':15} {'Wind':<12} {'Solar':<12} {'Load':<12}")
for i, name in enumerate(feature_names):
    print(f"{name:<15} {residual_corr[i, 0]:<12.4f} {residual_corr[i, 1]:<12.4f} {residual_corr[i, 2]:<12.4f}")

# 绘制残差相关性热力图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(residual_corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(feature_names)
ax.set_yticklabels(feature_names)

# 添加数值标注
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{residual_corr[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=12)

ax.set_title('残差相关性矩阵', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='相关系数')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: residual_correlation.png")

# ============================================================
# 3. Pearson 相关系数
# ============================================================
print("\n" + "="*60)
print("3. Pearson 相关系数分析")
print("="*60)

# 展平所有预测值和真实值
all_pred = {
    'Wind': test_pred[:, :, 0].flatten(),
    'Solar': test_pred[:, :, 1].flatten(),
    'Load': test_pred[:, :, 2].flatten()
}
all_true = {
    'Wind': test_true[:, :, 0].flatten(),
    'Solar': test_true[:, :, 1].flatten(),
    'Load': test_true[:, :, 2].flatten()
}

print("\n预测值 vs 真实值 Pearson 相关系数:")
print("-" * 50)
print(f"{'特征':<10} {'Pearson r':<15} {'p-value':<20}")
print("-" * 50)

pearson_results = {}
for name in feature_names:
    r, p = stats.pearsonr(all_pred[name], all_true[name])
    pearson_results[name] = {'r': r, 'p': p}
    print(f"{name:<10} {r:<15.6f} {p:<20.2e}")

print("-" * 50)

# 绘制预测值 vs 真实值散点图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(feature_names):
    # 随机采样10000个点避免图太密集
    n_samples = min(10000, len(all_pred[name]))
    idx = np.random.choice(len(all_pred[name]), n_samples, replace=False)
    
    axes[i].scatter(all_true[name][idx], all_pred[name][idx], alpha=0.3, s=5, c='steelblue')
    
    # 添加理想线 (y=x)
    min_val = min(all_true[name].min(), all_pred[name].min())
    max_val = max(all_true[name].max(), all_pred[name].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 (y=x)')
    
    # 添加回归线
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_true[name], all_pred[name])
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    axes[i].plot(x_line, y_line, 'g-', linewidth=2, label=f'回归线 (r={pearson_results[name]["r"]:.4f})')
    
    axes[i].set_title(f'{name} 预测值 vs 真实值', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('真实值', fontsize=12)
    axes[i].set_ylabel('预测值', fontsize=12)
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pred_vs_true_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: pred_vs_true_scatter.png")

# ============================================================
# 4. 综合分析报告
# ============================================================
print("\n" + "="*60)
print("4. 综合分析报告")
print("="*60)

print("\n【残差分布分析 - 用于DM模型场景生成】")
print("-" * 80)
for name in feature_names:
    stats_dict = residual_stats[name]
    print(f"\n{name}:")
    print(f"  均值: {stats_dict['mean']:.2f} (接近0表示无系统性偏差)")
    print(f"  标准差: {stats_dict['std']:.2f} (预测不确定性)")
    print(f"  偏度: {stats_dict['skewness']:.4f} (正偏=预测偏低, 负偏=预测偏高)")
    print(f"  峰度: {stats_dict['kurtosis']:.4f} (>0表示厚尾分布)")
    print(f"  正态性 p值: {stats_dict['normality_p']:.6f} (>0.05表示近似正态)")
    
    # 判断是否适合注入DM模型
    is_normal = stats_dict['normality_p'] > 0.05
    has_bias = abs(stats_dict['mean']) > stats_dict['std'] * 0.1
    
    if is_normal and not has_bias:
        recommendation = "[OK] 适合注入DM模型 - 残差近似正态分布且无系统性偏差"
    elif has_bias:
        recommendation = "[!!] 需要校正 - 存在系统性偏差，建议先修正预测模型"
    else:
        recommendation = "[!!] 需要变换 - 残差非正态，建议使用Copula或分位数回归"
    
    print(f"  建议: {recommendation}")

print("\n" + "="*60)
print("分析完成！所有图片已保存到:", output_dir)
print("="*60)
