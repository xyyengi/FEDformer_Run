import numpy as np
import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Solar 后处理：夜间负值置零
# ============================================================

print("="*60)
print("Solar 后处理：夜间负值置零")
print("="*60)

# 加载原始结果
results_dir = 'D:/FEDformer_Run/results/4.27'
splits = ['train', 'val', 'test']

# 创建后处理输出目录
output_dir = 'D:/FEDformer_Run/results/4.27/postprocessed'
os.makedirs(output_dir, exist_ok=True)

# 特征索引
SOLAR_IDX = 1  # Solar 在特征中的索引 (Wind=0, Solar=1, Load=2)

for split in splits:
    pred_path = os.path.join(results_dir, f'{split}_pred.npy')
    res_path = os.path.join(results_dir, f'{split}_res.npy')
    
    if not os.path.exists(pred_path):
        print(f"跳过 {split}，文件不存在")
        continue
    
    pred = np.load(pred_path)
    res = np.load(res_path)
    
    print(f"\n处理 {split} 集:")
    print(f"  原始 pred shape: {pred.shape}")
    print(f"  原始 res shape: {res.shape}")
    
    # 计算真实值
    true = pred + res
    
    # 统计原始Solar负值
    solar_pred_original = pred[:, :, SOLAR_IDX]
    solar_true = true[:, :, SOLAR_IDX]
    negative_count_before = np.sum(solar_pred_original < 0)
    negative_ratio_before = negative_count_before / solar_pred_original.size * 100
    
    print(f"  Solar 负值数量 (处理前): {negative_count_before} ({negative_ratio_before:.2f}%)")
    print(f"  Solar 预测范围 (处理前): [{solar_pred_original.min():.2f}, {solar_pred_original.max():.2f}]")
    
    # ============================================================
    # 后处理策略：Solar 预测值 < 0 时置零
    # ============================================================
    
    # 创建后处理后的预测值副本
    pred_postprocessed = pred.copy()
    
    # Solar 负值置零
    solar_mask = pred_postprocessed[:, :, SOLAR_IDX] < 0
    pred_postprocessed[:, :, SOLAR_IDX] = np.where(solar_mask, 0, pred_postprocessed[:, :, SOLAR_IDX])
    
    # 重新计算残差 (残差 = 预测值 - 真实值)
    residual_actual_minus_forecast_postprocessed = true - pred_postprocessed
    
    # 统计处理后Solar
    solar_pred_after = pred_postprocessed[:, :, SOLAR_IDX]
    negative_count_after = np.sum(solar_pred_after < 0)
    
    print(f"  Solar 负值数量 (处理后): {negative_count_after} (已全部置零)")
    print(f"  Solar 预测范围 (处理后): [{solar_pred_after.min():.2f}, {solar_pred_after.max():.2f}]")
    
    # 统计Solar残差变化
    solar_res_original = res[:, :, SOLAR_IDX]
    solar_res_after = residual_actual_minus_forecast_postprocessed[:, :, SOLAR_IDX]
    
    print(f"  Solar 残差均值 (处理前): {solar_res_original.mean():.2f}")
    print(f"  Solar 残差均值 (处理后): {solar_res_after.mean():.2f}")
    print(f"  Solar 残差标准差 (处理前): {solar_res_original.std():.2f}")
    print(f"  Solar 残差标准差 (处理后): {solar_res_after.std():.2f}")
    
    # 保存后处理结果
    np.save(os.path.join(output_dir, f'{split}_pred.npy'), pred_postprocessed)
    np.save(os.path.join(output_dir, f'{split}_res.npy'), residual_actual_minus_forecast_postprocessed)
    np.save(os.path.join(output_dir, f'{split}_true.npy'), true)  # 真实值不变
    
    print(f"  已保存: {split}_pred.npy, {split}_res.npy, {split}_true.npy")

# ============================================================
# 可视化对比
# ============================================================
print("\n" + "="*60)
print("生成可视化对比图")
print("="*60)

# 只对测试集生成对比图
pred_original = np.load(os.path.join(results_dir, 'test_pred.npy'))
res_original = np.load(os.path.join(results_dir, 'test_res.npy'))
pred_postprocessed = np.load(os.path.join(output_dir, 'test_pred.npy'))
res_postprocessed = np.load(os.path.join(output_dir, 'test_res.npy'))
true = np.load(os.path.join(output_dir, 'test_true.npy'))

# 取最后168小时对比
last_idx = -1
hours = np.arange(1, 169)

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# Solar 原始预测
axes[0].plot(hours, true[last_idx, :, SOLAR_IDX], 'b-', linewidth=2, label='真实值', marker='o', markersize=3)
axes[0].plot(hours, pred_original[last_idx, :, SOLAR_IDX], 'r--', linewidth=2, label='原始预测值', marker='x', markersize=3)
axes[0].fill_between(hours, pred_original[last_idx, :, SOLAR_IDX], true[last_idx, :, SOLAR_IDX], 
                     alpha=0.3, color='red', label='原始误差')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].set_title('Solar - 原始预测 (含负值)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('小时 (h)', fontsize=12)
axes[0].set_ylabel('出力 (MW)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, 168)

# Solar 后处理预测
axes[1].plot(hours, true[last_idx, :, SOLAR_IDX], 'b-', linewidth=2, label='真实值', marker='o', markersize=3)
axes[1].plot(hours, pred_postprocessed[last_idx, :, SOLAR_IDX], 'g-', linewidth=2, label='后处理预测值', marker='s', markersize=3)
axes[1].fill_between(hours, pred_postprocessed[last_idx, :, SOLAR_IDX], true[last_idx, :, SOLAR_IDX], 
                     alpha=0.3, color='green', label='后处理误差')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_title('Solar - 后处理预测 (负值置零)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('小时 (h)', fontsize=12)
axes[1].set_ylabel('出力 (MW)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(1, 168)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'solar_postprocess_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: solar_postprocess_comparison.png")

# ============================================================
# 统计汇总
# ============================================================
print("\n" + "="*60)
print("后处理统计汇总")
print("="*60)

print("\n【Solar 后处理效果】")
print("-" * 80)

for split in splits:
    try:
        res_original = np.load(os.path.join(results_dir, f'{split}_res.npy'))
        res_postprocessed = np.load(os.path.join(output_dir, f'{split}_res.npy'))
        
        solar_res_orig = res_original[:, :, SOLAR_IDX].flatten()
        solar_res_post = res_postprocessed[:, :, SOLAR_IDX].flatten()
        
        print(f"\n{split} 集 Solar 残差:")
        print(f"  原始: 均值={solar_res_orig.mean():.2f}, 标准差={solar_res_orig.std():.2f}")
        print(f"  后处理: 均值={solar_res_post.mean():.2f}, 标准差={solar_res_post.std():.2f}")
        
        # MAE/RMSE 对比
        mae_orig = np.mean(np.abs(solar_res_orig))
        rmse_orig = np.sqrt(np.mean(solar_res_orig**2))
        mae_post = np.mean(np.abs(solar_res_post))
        rmse_post = np.sqrt(np.mean(solar_res_post**2))
        
        print(f"  MAE: {mae_orig:.2f} -> {mae_post:.2f} (改善: {(mae_orig-mae_post)/mae_orig*100:.1f}%)")
        print(f"  RMSE: {rmse_orig:.2f} -> {rmse_post:.2f} (改善: {(rmse_orig-rmse_post)/rmse_orig*100:.1f}%)")
    except:
        pass

print("\n" + "="*60)
print(f"后处理完成！文件已保存到: {output_dir}")
print("="*60)

print("\n【下一步使用建议】")
print("将 postprocessed 目录下的文件用于 DM 模型:")
print(f"  - {output_dir}/train_pred.npy, train_res.npy")
print(f"  - {output_dir}/val_pred.npy, val_res.npy")
print(f"  - {output_dir}/test_pred.npy, test_res.npy")
