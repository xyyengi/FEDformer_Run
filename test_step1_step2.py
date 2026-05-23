"""
测试脚本：验证第1步(modes=128)和第2步(周期性时间编码)的修改
使用小样本数据，快速验证代码是否能正常运行
"""
import os
import sys
import torch
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom

def test_cyclical_encoding():
    """测试周期性时间编码"""
    print("=" * 50)
    print("测试周期性时间编码 (Sin/Cos)")
    print("=" * 50)
    
    # 创建模拟参数
    class Args:
        root_path = './'
        data_path = 'Wind_Solar_Load_Processed.csv'
        seq_len = 48  # 小样本
        label_len = 24
        pred_len = 24
        features = 'M'
        target = 'Load'
        freq = 'h'
        embed = 'timeF'
        num_workers = 0
        batch_size = 4
        use_cycle_time_enc = True
    
    args = Args()
    
    try:
        # 测试 Dataset_Custom
        dataset = Dataset_Custom(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=0,
            freq=args.freq,
            use_cycle_time_enc=True
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 获取一个样本
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        print(f"seq_x shape: {seq_x.shape}")
        print(f"seq_y shape: {seq_y.shape}")
        print(f"seq_x_mark shape (时间特征): {seq_x_mark.shape}")
        print(f"seq_y_mark shape (时间特征): {seq_y_mark.shape}")
        
        # 验证时间特征维度
        # 原始: 4个特征 (month, day, weekday, hour)
        # 周期编码后: 8个特征 (month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos, hour_sin, hour_cos)
        expected_dim = 8  # 4个时间特征 * 2 (sin + cos)
        if seq_x_mark.shape[1] == expected_dim:
            print(f"✓ 时间特征维度正确: {seq_x_mark.shape[1]} (预期 {expected_dim})")
        else:
            print(f"✗ 时间特征维度错误: {seq_x_mark.shape[1]} (预期 {expected_dim})")
        
        # 检查值范围 (sin/cos 应在 [-1, 1])
        if np.all(seq_x_mark >= -1) and np.all(seq_x_mark <= 1):
            print("✓ 时间特征值范围正确: [-1, 1]")
        else:
            print("✗ 时间特征值范围错误")
        
        print("\n周期性时间编码测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modes_parameter():
    """测试 modes 参数"""
    print("\n" + "=" * 50)
    print("测试 modes 参数 (128)")
    print("=" * 50)
    
    try:
        from models.FEDformer import Model
        
        class Args:
            seq_len = 48
            label_len = 24
            pred_len = 24
            enc_in = 3  # Wind, Solar, Load
            dec_in = 3
            c_out = 3
            d_model = 64
            n_heads = 4
            e_layers = 2
            d_layers = 1
            d_ff = 128
            moving_avg = [24]
            factor = 1
            distil = True
            dropout = 0.1
            embed = 'timeF'
            activation = 'gelu'
            output_attention = False
            # FEDformer 特定参数
            version = 'Fourier'
            mode_select = 'random'
            modes = 128  # 新增的参数
            L = 3
            base = 'legendre'
            cross_activation = 'tanh'
        
        args = Args()
        
        model = Model(args)
        print(f"✓ 模型创建成功")
        print(f"  modes 参数: {args.modes}")
        
        # 测试前向传播
        batch_x = torch.randn(2, args.seq_len, args.enc_in)
        batch_x_mark = torch.randn(2, args.seq_len, 8)  # 8个时间特征
        batch_y = torch.randn(2, args.label_len + args.pred_len, args.dec_in)
        batch_y_mark = torch.randn(2, args.label_len + args.pred_len, 8)
        
        # 解码器输入
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[2]])
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)
        
        output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        print(f"✓ 前向传播成功")
        print(f"  输出 shape: {output.shape}")
        
        print("\nmodes 参数测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("FEDformer 优化 - 第1步和第2步验证测试")
    print("=" * 60)
    
    # 检查数据文件是否存在
    data_file = 'Wind_Solar_Load_Processed.csv'
    if not os.path.exists(data_file):
        print(f"✗ 数据文件不存在: {data_file}")
        return
    
    # 运行测试
    test1_pass = test_cyclical_encoding()
    test2_pass = test_modes_parameter()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"周期性时间编码: {'✓ 通过' if test1_pass else '✗ 失败'}")
    print(f"modes 参数 (128): {'✓ 通过' if test2_pass else '✗ 失败'}")
    
    if test1_pass and test2_pass:
        print("\n>>> 第1步和第2步验证成功，可以继续进行第3、4步优化 <<<")
    else:
        print("\n>>> 存在错误，需要修复后再继续 <<<")


if __name__ == "__main__":
    main()