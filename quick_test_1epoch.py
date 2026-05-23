"""
快速测试：只训练 1 个 epoch，检查 Solar 是否还有负值
"""
import os
import numpy as np
import torch
import argparse
from data_provider.data_factory import data_provider
from models import FEDformer
from exp.exp_main import Exp_Main

# 创建参数
class Args:
    def __init__(self):
        self.model = 'FEDformer'
        self.root_path = './'
        self.data_path = 'Wind_Solar_Load_Processed.csv'
        self.features = 'M'
        self.target = 'Load'
        self.freq = 'h'
        self.checkpoints = './checkpoints'
        
        # 序列长度
        self.seq_len = 336
        self.label_len = 168
        self.pred_len = 168
        
        # 模型参数
        self.enc_in = 11
        self.dec_in = 11
        self.c_out = 11
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 3
        self.dropout = 0.05
        self.embed = 'timeF'
        self.distil = True
        self.activation = 'gelu'
        self.output_attention = False
        self.num_time_features = 8
        self.use_cycle_time_enc = True
        
        # FEDformer 特定参数
        self.modes = 64
        self.mode_select = 'random'
        self.version = 'Fourier'
        
        # 训练参数
        self.train_epochs = 1  # 只训练 1 个 epoch
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.loss = 'mse'
        self.patience = 3
        self.use_gpu = True if torch.cuda.is_available() else False
        self.use_multi_gpu = False
        self.devices = '0'
        self.device_ids = [0]
        self.use_amp = False
        self.num_workers = 0
        self.itr = 1
        self.des = 'Test'
        self.topk = 5
        self.solar_idx = 1

args = Args()

print("=" * 60)
print("快速测试：训练 1 个 epoch 检查负值问题")
print("=" * 60)

# 获取数据
print("\n1. 加载训练数据...")
train_data, train_loader = data_provider(args, flag='train')
print(f"   训练数据形状: {train_data.data_x.shape}")

# 检查 scaler 参数
print("\n2. 检查 Scaler 参数...")
if hasattr(train_data, 'scaler'):
    print(f"   Scaler 类型: {type(train_data.scaler).__name__}")
    if hasattr(train_data.scaler, 'data_min_'):
        print(f"   data_min_: {train_data.scaler.data_min_}")
        print(f"   data_max_: {train_data.scaler.data_max_}")
        print(f"   ✅ 使用 MinMaxScaler")
    elif hasattr(train_data.scaler, 'mean_'):
        print(f"   mean_: {train_data.scaler.mean_}")
        print(f"   ❌ 使用 StandardScaler (会产生负值!)")

# 快速训练 1 个 epoch
print("\n3. 快速训练 1 个 epoch...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   使用设备: {device}")

# 构建模型
model = FEDformer.Model(args).float().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.MSELoss()

model.train()
batch_count = 0
for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    
    # decoder input
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    
    # forward
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # 只取预测部分
    f_dim = 0  # M 模式
    batch_y = batch_y[:, -args.pred_len:, f_dim:]
    
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    batch_count += 1
    if batch_count >= 10:  # 只训练 10 个 batch
        break

print(f"   训练了 {batch_count} 个 batch")

# 测试预测
print("\n4. 测试预测结果...")
test_data, test_loader = data_provider(args, flag='test')

model.eval()
preds = []
trues = []

with torch.no_grad():
    batch_count = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
        
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        f_dim = 0
        batch_y = batch_y[:, -args.pred_len:, f_dim:]
        
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y.cpu().numpy())
        
        batch_count += 1
        if batch_count >= 5:  # 只测试 5 个 batch
            break

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

print(f"   预测形状: {preds.shape}")

# 反归一化
print("\n5. 反归一化...")
if hasattr(test_data, 'inverse_transform'):
    preds_inv = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    trues_inv = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)
    print("   ✅ 已应用反归一化")
else:
    preds_inv = preds
    trues_inv = trues
    print("   ❌ 没有 inverse_transform 方法")

# 检查负值
print("\n6. 检查 Solar 负值...")
solar_pred = preds_inv[:, :, 1]  # Solar 是第 2 个特征
solar_true = trues_inv[:, :, 1]

neg_count = np.sum(solar_pred < 0)
total = solar_pred.size
neg_pct = neg_count / total * 100

print(f"   Solar 预测值范围: [{solar_pred.min():.2f}, {solar_pred.max():.2f}]")
print(f"   Solar 真实值范围: [{solar_true.min():.2f}, {solar_true.max():.2f}]")
print(f"   负值数量: {neg_count} / {total} ({neg_pct:.2f}%)")

if neg_count == 0:
    print("\n" + "=" * 60)
    print("✅ 成功！Solar 没有负值！")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("❌ 问题仍然存在！Solar 仍有负值！")
    print("=" * 60)
    print("\n可能的原因:")
    print("1. exp_main.py 没有同步到服务器")
    print("2. data_loader.py 的 inverse_transform 方法有问题")
    print("3. scaler 参数保存/加载方式不正确")

# 保存测试结果
np.save('quick_test_pred.npy', preds_inv)
np.save('quick_test_true.npy', trues_inv)
print("\n结果已保存到 quick_test_pred.npy 和 quick_test_true.npy")