# ============================================================
# 编码器-解码器与分解机制 - 详细中文注释版
# ============================================================
# 
# 【本文件核心内容】
# 1. 序列分解（Series Decomposition）- 核心创新
# 2. 编码器（Encoder）和解码器（Decoder）
# 3. 编码器层和解码器层
# 
# 【序列分解 - FEDformer的关键创新】
# 
# 什么是序列分解？
# -------------------
# 将时间序列分解为两个部分：
#   原始序列 = 趋势项 (Trend) + 季节项 (Seasonal)
# 
# 趋势项：代表长期变化方向，变化缓慢、平滑
# 季节项：代表周期性波动，变化快速、有规律
# 
# 为什么需要分解？
# -------------------
# 1. 趋势和季节有不同的特性，分开处理更有效
# 2. 趋势项适合低频建模（变化慢）
# 3. 季节项适合高频建模（变化快）
# 4. 分解使模型更稳定，训练更容易
# 
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.SelfAttention_Family import FullAttention


class my_Layernorm(nn.Module):
    """
    ============================================================
    专门为季节项设计的层归一化
    ============================================================
    季节项是周期性波动，标准LayerNorm可能破坏周期性
    改进：在LayerNorm后减去均值，保持周期特性
    ============================================================
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    ============================================================
    移动平均 - 提取趋势项的核心操作
    ============================================================
    
    【原理】用窗口内数据的平均值代替当前值
    消除短期波动，保留长期趋势
    
    【参数】
    - kernel_size: 窗口大小（越大趋势越平滑）
    - stride: 步长（通常为1）
    
    【窗口大小影响】
    - 较大值：趋势更平滑，过滤更多细节
    - 较小值：保留更多细节，趋势变化更快
    ============================================================
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 边界填充：用首尾值重复填充，保持序列长度
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        
        # 平均池化：[B, L, D] -> [B, D, L] -> avg -> [B, L, D]
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    ============================================================
    单尺度序列分解
    ============================================================
    
    【数学公式】
    trend = moving_avg(x)  # 移动平均提取趋势
    seasonal = x - trend   # 原始减去趋势得到季节
    
    【图示】
    原始序列:  /\_/\_/\_/\_/\_/
    趋势项:   __________________  (平滑的线)
    季节项:   /\_/\_/\_/\_/\_/  (周期性波动)
    ============================================================
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 趋势项
        res = x - moving_mean              # 季节项
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    ============================================================
    多尺度序列分解
    ============================================================
    
    【功能】使用多个窗口大小进行分解，加权组合
    
    【为什么多尺度？】
    不同窗口捕捉不同时间尺度的趋势：
    - 小窗口：捕捉短期趋势
    - 大窗口：捕捉长期趋势
    
    【示例】kernel_size = [12, 24]
    - 12小时窗口：捕捉半天趋势
    - 24小时窗口：捕捉一天趋势
    ============================================================
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))  # 可学习权重

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        # Softmax加权组合
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    ============================================================
    编码器层 - 渐进式分解架构
    ============================================================
    
    【工作流程】
    1. 频域自注意力 → 残差连接
    2. 第一次分解（提取趋势）
    3. 前馈网络（Conv1D）
    4. 第二次分解（进一步精炼）
    
    【渐进式分解的优势】
    - 每层都提取趋势，逐步去除噪声
    - 季节项在层间传递，保留周期性信息
    ============================================================
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        # 前馈网络：两个1D卷积
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        # 分解模块
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 1. 自注意力 + 残差连接
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 2. 第一次分解
        x, _ = self.decomp1(x)
        y = x
        
        # 3. 前馈网络
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 4. 第二次分解
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    ============================================================
    编码器 - 多层EncoderLayer堆叠
    ============================================================
    
    【功能】处理历史数据，提取特征，生成编码表示
    
    【结构】
    - 多个EncoderLayer顺序堆叠
    - 最后有层归一化
    ============================================================
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    ============================================================
    解码器层 - 渐进式分解架构
    ============================================================
    
    【工作流程】
    1. 自注意力 → 分解（累积趋势1）
    2. 交叉注意力（与编码器交互）→ 分解（累积趋势2）
    3. 前馈网络 → 分解（累积趋势3）
    4. 返回季节项和累积趋势项
    
    【关键特点】
    - 每步都分解，趋势项被累积保存
    - 季节项继续传递处理
    - 最终预测 = 季节项 + 累积趋势项
    ============================================================
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # 前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        # 三次分解（对应三个操作）
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        # 趋势项投影：将趋势映射到输出维度
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. 自注意力 + 分解
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        
        # 2. 交叉注意力 + 分解
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        
        # 3. 前馈网络 + 分解
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        # 4. 累积趋势项并投影
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    ============================================================
    解码器 - 多层DecoderLayer堆叠
    ============================================================
    
    【功能】结合历史信息和编码器输出，生成预测
    
    【结构】
    - 多个DecoderLayer顺序堆叠
    - 层归一化
    - 输出投影
    
    【输入】
    - x: 解码器输入（季节项初始化）
    - cross: 编码器输出
    - trend: 初始趋势项
    ============================================================
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # 累积趋势项
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("分解机制测试")
    print("=" * 60)
    
    # 测试移动平均
    x = torch.randn(2, 100, 4)  # [batch, seq_len, features]
    
    print("\n--- 移动平均测试 ---")
    ma = moving_avg(kernel_size=25, stride=1)
    trend = ma(x)
    print(f"输入形状: {x.shape}")
    print(f"趋势形状: {trend.shape}")
    
    # 测试序列分解
    print("\n--- 序列分解测试 ---")
    decomp = series_decomp(kernel_size=25)
    seasonal, trend = decomp(x)
    print(f"季节项形状: {seasonal.shape}")
    print(f"趋势项形状: {trend.shape}")
    
    # 验证分解正确性
    reconstructed = seasonal + trend
    diff = torch.abs(x - reconstructed).max()
    print(f"重建误差（应该接近0）: {diff.item():.6f}")
    
    print("\n测试完成！")