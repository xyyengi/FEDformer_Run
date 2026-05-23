"""
训练和预测日志系统
每次运行生成独立的日志文件夹，包含：
- 训练日志 (loss曲线、学习率等)
- 预测结果 (指标、可视化曲线)
- 残差分析
"""
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


class ExperimentLogger:
    """实验日志管理器"""
    
    def __init__(self, base_dir='./experiments', exp_name=None):
        """
        初始化日志器
        
        Args:
            base_dir: 日志根目录
            exp_name: 实验名称，如果为None则自动生成时间戳名称
        """
        self.base_dir = base_dir
        self.exp_name = exp_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(base_dir, self.exp_name)
        self.train_log = []
        self.metrics = {}
        
        # 创建目录结构
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'predictions'), exist_ok=True)
        
    def log_args(self, args):
        """记录参数配置"""
        args_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
        args_path = os.path.join(self.exp_dir, 'config.json')
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=2, default=str)
        print(f"[Logger] 配置已保存到 {args_path}")
    
    def log_train_step(self, epoch, train_loss, val_loss, lr=None):
        """记录训练步骤"""
        log_entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(lr) if lr else None,
            'timestamp': datetime.now().isoformat()
        }
        self.train_log.append(log_entry)
        
        # 实时写入
        self._save_train_log()
    
    def _save_train_log(self):
        """保存训练日志"""
        log_path = os.path.join(self.exp_dir, 'train_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        losses = np.array([
            [
                log['epoch'],
                log['train_loss'],
                log['val_loss'],
                log['lr'] if log['lr'] is not None else np.nan,
            ]
            for log in self.train_log
        ], dtype=float)
        np.save(os.path.join(self.exp_dir, 'losses.npy'), losses)
    
    def log_metrics(self, metrics_dict, split='test'):
        """记录评估指标"""
        self.metrics[split] = metrics_dict
        metrics_path = os.path.join(self.exp_dir, f'metrics_{split}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"[Logger] {split} 指标: {metrics_dict}")
    
    def save_predictions(self, preds, trues, split='test', feature_names=None):
        """
        保存预测结果并生成可视化
        
        Args:
            preds: 预测值 (samples, pred_len, features)
            trues: 真实值 (samples, pred_len, features)
            split: 数据集划分 (train/val/test)
            feature_names: 特征名称列表
        """
        # 保存numpy数组
        pred_dir = os.path.join(self.exp_dir, 'predictions')
        np.save(os.path.join(pred_dir, f'{split}_pred.npy'), preds)
        np.save(os.path.join(pred_dir, f'{split}_true.npy'), trues)
        
        # 计算残差
        residual_actual_minus_forecast = trues - preds
        np.save(os.path.join(pred_dir, f'{split}_residual.npy'), residual_actual_minus_forecast)
        
        # 生成可视化
        self._plot_predictions(preds, trues, split, feature_names)
        self._plot_residuals(residual_actual_minus_forecast, split, feature_names)
        
        print(f"[Logger] {split} 预测结果已保存到 {pred_dir}")
    
    def _plot_predictions(self, preds, trues, split, feature_names=None):
        """绘制预测对比图"""
        n_features = preds.shape[-1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 确保特征数量匹配
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        elif len(feature_names) < n_features:
            feature_names = list(feature_names) + [f'Feature_{i}' for i in range(len(feature_names), n_features)]
        
        fig_dir = os.path.join(self.exp_dir, 'figures')
        
        # 选择几个样本进行可视化 (第一个、中间、最后一个)
        n_samples = preds.shape[0]
        sample_indices = [0, n_samples // 2, n_samples - 1]
        
        for feat_idx, feat_name in enumerate(feature_names):
            fig, axes = plt.subplots(len(sample_indices), 1, figsize=(15, 4 * len(sample_indices)))
            if len(sample_indices) == 1:
                axes = [axes]
            
            for ax_idx, sample_idx in enumerate(sample_indices):
                ax = axes[ax_idx]
                pred = preds[sample_idx, :, feat_idx]
                true = trues[sample_idx, :, feat_idx]
                
                ax.plot(true, label='True', linewidth=2, color='blue')
                ax.plot(pred, label='Pred', linewidth=2, color='red', alpha=0.7)
                ax.fill_between(range(len(true)), true, pred, alpha=0.3, color='gray')
                ax.set_title(f'{feat_name} - Sample {sample_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{split}_{feat_name}_comparison.pdf'))
            plt.close()
    
    def _plot_residuals(self, residuals, split, feature_names=None):
        """绘制残差分析图"""
        n_features = residuals.shape[-1]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 确保特征数量匹配
        if len(feature_names) > n_features:
            feature_names = feature_names[:n_features]
        elif len(feature_names) < n_features:
            feature_names = list(feature_names) + [f'Feature_{i}' for i in range(len(feature_names), n_features)]
        
        fig_dir = os.path.join(self.exp_dir, 'figures')
        
        for feat_idx, feat_name in enumerate(feature_names):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 展平所有样本的残差
            res_flat = residuals[:, :, feat_idx].flatten()
            
            # 1. 残差分布直方图
            axes[0, 0].hist(res_flat, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title(f'{feat_name} - Residual Distribution')
            axes[0, 0].set_xlabel('Residual')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            # 2. 残差时序图
            axes[0, 1].plot(res_flat, alpha=0.7)
            axes[0, 1].set_title(f'{feat_name} - Residual Time Series')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Residual')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            
            # 3. 残差箱线图 (按预测步长)
            pred_len = residuals.shape[1]
            res_by_step = [residuals[:, t, feat_idx] for t in range(pred_len)]
            axes[1, 0].boxplot(res_by_step, positions=range(pred_len))
            axes[1, 0].set_title(f'{feat_name} - Residual by Prediction Step')
            axes[1, 0].set_xlabel('Prediction Step')
            axes[1, 0].set_ylabel('Residual')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
            
            # 4. 残差统计信息
            mean_res = np.mean(res_flat)
            std_res = np.std(res_flat)
            mae_res = np.mean(np.abs(res_flat))
            rmse_res = np.sqrt(np.mean(res_flat ** 2))
            
            stats_text = f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}\nMAE: {mae_res:.4f}\nRMSE: {rmse_res:.4f}'
            axes[1, 1].text(0.5, 0.5, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=14, verticalalignment='center', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].set_title(f'{feat_name} - Residual Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{split}_{feat_name}_residual_analysis.pdf'))
            plt.close()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_log:
            print("[Logger] 没有训练日志")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = [log['epoch'] for log in self.train_log]
        train_losses = [log['train_loss'] for log in self.train_log]
        val_losses = [log['val_loss'] for log in self.train_log]
        
        # Loss曲线
        axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 学习率曲线
        lrs = [log['lr'] for log in self.train_log if log['lr']]
        if lrs:
            lr_epochs = [log['epoch'] for log in self.train_log if log['lr']]
            axes[1].plot(lr_epochs, lrs, label='Learning Rate', linewidth=2, color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figure_base = os.path.join(self.exp_dir, 'figures', 'training_curves')
        plt.savefig(figure_base + '.pdf')
        plt.savefig(figure_base + '.png', dpi=160)
        plt.close()
        print(f"[Logger] 训练曲线已保存")
    
    def generate_summary_report(self):
        """生成实验总结报告"""
        report_path = os.path.join(self.exp_dir, 'summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"实验报告: {self.exp_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 训练日志摘要\n")
            f.write("-" * 40 + "\n")
            if self.train_log:
                f.write(f"总轮数: {len(self.train_log)}\n")
                f.write(f"最终训练损失: {self.train_log[-1]['train_loss']:.6f}\n")
                f.write(f"最终验证损失: {self.train_log[-1]['val_loss']:.6f}\n")
                best_val = min(self.train_log, key=lambda x: x['val_loss'])
                f.write(f"最佳验证损失: {best_val['val_loss']:.6f} (Epoch {best_val['epoch']})\n")
            f.write("\n")
            
            f.write("2. 评估指标\n")
            f.write("-" * 40 + "\n")
            for split, metrics in self.metrics.items():
                f.write(f"\n{split}:\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v:.6f}\n")
            
            f.write("\n3. 文件列表\n")
            f.write("-" * 40 + "\n")
            for root, dirs, files in os.walk(self.exp_dir):
                for file in files:
                    f.write(f"  {os.path.join(root, file)}\n")
        
        print(f"[Logger] 实验报告已保存到 {report_path}")
