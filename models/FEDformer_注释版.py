# ============================================================
# FEDformer 模型 - 详细中文注释版
# ============================================================
# 
# 【模型概述】
# FEDformer (Frequency Enhanced Decomposed Transformer) 是一种
# 专门用于时间序列预测的深度学习模型。
# 
# 【两大核心创新】
# 1. 频域注意力：在频率域进行注意力计算，复杂度从O(N²)降到O(N)
# 2. 分解架构：将序列分解为趋势项和季节项，分别处理
# 
# 【适用场景】
# - 长序列时间序列预测
# - 多变量预测（如电力负荷、风速、太阳能等）
# - 具有周期性模式的数据
# 
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    ============================================================
    FEDformer 主模型类
    ============================================================
    
    【功能】整合所有组件，完成时间序列预测
    
    【输入输出】
    输入：
        - x_enc: 编码器输入，历史时间序列数据 [batch, seq_len, features]
        - x_mark_enc: 编码器时间标记（时间特征）[batch, seq_len, time_features]
        - x_dec: 解码器输入（部分历史+待预测位置）
        - x_mark_dec: 解码器时间标记
    
    输出：
        - 预测的未来时间序列 [batch, pred_len, features]
    
    【关键参数说明】
    - version: 'Fourier' 或 'Wavelets'，选择频域方法
    - modes: 保留的频率分量数量（影响计算量和精度）
    - seq_len: 输入序列长度（历史数据长度）
    - label_len: 解码器已知部分长度
    - pred_len: 预测长度
    - moving_avg: 分解时移动平均的窗口大小
    ============================================================
    """
    
    def __init__(self, configs):
        """
        ============================================================
        模型初始化 - 构建所有组件
        ============================================================
        
        【初始化流程】
        1. 设置基本参数
        2. 创建序列分解模块
        3. 创建嵌入层
        4. 选择并创建频域注意力模块
        5. 创建编码器和解码器
        ============================================================
        """
        super(Model, self).__init__()
        
        # ============================================================
        # 第一步：设置基本参数
        # ============================================================
        
        # version: 选择频域方法
        # - 'Fourier': 傅里叶变换，适合有明显周期性的数据
        # - 'Wavelets': 小波变换，适合有突变或多尺度模式的数据
        self.version = configs.version
        
        # mode_select: 频率选择方法
        # - 'random': 随机选择频率分量
        # - 'lowest': 选择最低频率（通常包含主要趋势信息）
        self.mode_select = configs.mode_select
        
        # modes: 保留的频率分量数量
        # - 较大值：保留更多信息，计算量更大
        # - 较小值：过滤更多噪声，计算量更小
        # - 推荐：根据序列长度设置，一般为 seq_len/4 到 seq_len/2
        self.modes = configs.modes
        
        # 序列长度参数
        self.seq_len = configs.seq_len      # 输入序列长度（历史数据）
        self.label_len = configs.label_len  # 解码器输入的已知部分长度
        self.pred_len = configs.pred_len    # 预测长度（未来数据）
        
        # 是否输出注意力权重（用于可视化分析）
        self.output_attention = configs.output_attention

        # ============================================================
        # 第二步：创建序列分解模块
        # ============================================================
        # 
        # 【什么是序列分解？】
        # 将时间序列分解为两部分：
        # - 趋势项 (Trend): 长期变化方向，通过移动平均提取
        # - 季节项 (Seasonal): 周期性波动，原始序列减去趋势项
        # 
        # 【为什么需要分解？】
        # - 趋势和季节有不同的特性，分开处理更有效
        # - 趋势项变化缓慢，适合低频建模
        # - 季节项变化快速，适合高频建模
        # ============================================================
        
        kernel_size = configs.moving_avg  # 移动平均窗口大小
        
        if isinstance(kernel_size, list):
            # 多尺度分解：使用多个不同大小的窗口
            # 例如 [12, 24] 表示同时使用12和24作为窗口
            # 可以捕捉不同时间尺度的趋势
            self.decomp = series_decomp_multi(kernel_size)
        else:
            # 单尺度分解：使用固定大小的窗口
            self.decomp = series_decomp(kernel_size)

        # ============================================================
        # 第三步：创建嵌入层
        # ============================================================
        # 
        # 【嵌入层的作用】
        # 将原始数据转换为模型可以处理的特征表示
        # 
        # 【FEDformer的特殊之处】
        # 使用 DataEmbedding_wo_pos（无位置编码）
        # 原因：频域注意力天然具有全局感受野，不需要位置编码
        # 
        # 【嵌入层组成】
        # 1. TokenEmbedding: 值嵌入，通过1D卷积提取特征
        # 2. TemporalEmbedding: 时间嵌入，编码时间信息
        # ============================================================
        
        # 编码器嵌入层
        # enc_in: 输入特征数量（如 Wind, Solar, Load = 3）
        # d_model: 模型维度（特征扩展后的维度，如512）
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,      # 输入特征维度
            configs.d_model,     # 模型维度
            configs.embed,       # 嵌入类型
            configs.freq,        # 时间频率（'h'表示小时）
            configs.dropout      # dropout率
        )
        
        # 解码器嵌入层
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # ============================================================
        # 第四步：选择并创建频域注意力模块
        # ============================================================
        # 
        # 【频域注意力的核心思想】
        # 1. 将时域信号转换到频域（傅里叶/小波变换）
        # 2. 在频域进行注意力计算
        # 3. 转换回时域输出
        # 
        # 【为什么在频域做注意力？】
        # - 计算效率：复杂度从O(N²)降到O(N)
        # - 物理意义：时间序列的重要信息集中在特定频率
        # - 全局感受野：频域每个点包含整个序列信息
        # ============================================================
        
        if configs.version == 'Wavelets':
            # ============================================================
            # 小波版本 - MultiWaveletTransform
            # ============================================================
            # 
            # 【小波变换的特点】
            # - 多尺度分析：同时捕捉全局和局部模式
            # - 保留时间信息：知道"什么频率在什么时候出现"
            # - 适合非平稳信号：有突变或局部特征的数据
            # ============================================================
            
            # 编码器自注意力：小波变换
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model,  # 输入通道数
                L=configs.L,          # 小波层数
                base=configs.base     # 小波基函数类型
            )
            
            # 解码器自注意力：小波变换
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model,
                L=configs.L,
                base=configs.base
            )
            
            # 解码器交叉注意力：小波交叉注意力
            # 【交叉注意力】Query来自解码器，Key/Value来自编码器
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,  # Query序列长度
                seq_len_kv=self.seq_len,                       # Key/Value序列长度
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation
            )
        else:
            # ============================================================
            # 傅里叶版本 - FourierBlock
            # ============================================================
            # 
            # 【傅里叶变换的特点】
            # - 全局频率分析：捕捉整体周期性
            # - 计算快速：FFT算法效率高
            # - 适合平稳信号：有明显周期性的数据
            # ============================================================
            
            # 编码器自注意力：傅里叶块
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,              # 保留的频率分量数
                mode_select_method=configs.mode_select  # 频率选择方法
            )
            
            # 解码器自注意力：傅里叶块
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len//2 + self.pred_len,  # 解码器序列长度
                modes=configs.modes,
                mode_select_method=configs.mode_select
            )
            
            # 解码器交叉注意力：傅里叶交叉注意力
            # 【交叉注意力作用】让解码器"查询"编码器的信息
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len//2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select
            )
        
        # ============================================================
        # 第五步：创建编码器和解码器
        # ============================================================
        # 
        # 【编码器作用】
        # - 处理历史数据，提取特征
        # - 生成编码表示，供解码器使用
        # 
        # 【解码器作用】
        # - 结合历史信息和编码器输出
        # - 生成预测结果
        # 
        # 【渐进式分解架构】
        # 每一层都进行分解，逐步精炼趋势和季节
        # ============================================================
        
        # 计算实际使用的频率模式数量
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        # 编码器：多层EncoderLayer堆叠
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 注意力层：使用AutoCorrelationLayer包装频域注意力
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, 
                        configs.n_heads  # 注意力头数
                    ),
                    configs.d_model,
                    configs.d_ff,         # 前馈网络维度
                    moving_avg=configs.moving_avg,  # 分解窗口
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # 编码器层数
            ],
            norm_layer=my_Layernorm(configs.d_model)  # 层归一化
        )
        
        # 解码器：多层DecoderLayer堆叠
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # 自注意力：解码器内部的注意力
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, 
                        configs.n_heads
                    ),
                    # 交叉注意力：解码器与编码器之间的注意力
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.c_out,        # 输出特征数
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)  # 解码器层数
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)  # 输出投影
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        ============================================================
        前向传播 - 数据如何流过模型
        ============================================================
        
        【完整数据流】
        1. 初始分解：将输入分解为趋势和季节
        2. 准备解码器输入
        3. 编码器处理
        4. 解码器处理
        5. 合并输出
        
        【输入形状说明】
        - x_enc: [batch_size, seq_len, features] 例如 [32, 336, 3]
        - x_mark_enc: [batch_size, seq_len, time_features] 例如 [32, 336, 8]
        ============================================================
        """
        
        # ============================================================
        # 第一步：初始分解
        # ============================================================
        # 
        # 【目的】在进入模型前，先分解输入序列
        # 这样解码器可以分别处理趋势和季节
        # ============================================================
        
        # 计算历史数据的均值，用于初始化预测部分
        # mean: [batch, pred_len, features]
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        
        # 创建零张量，用于填充预测部分
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)
        
        # 序列分解：得到季节项和趋势项
        # seasonal_init: 周期性波动部分
        # trend_init: 长期趋势部分
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # ============================================================
        # 第二步：准备解码器输入
        # ============================================================
        # 
        # 【解码器输入结构】
        # 解码器需要两部分输入：
        # 1. 已知的历史部分 (label_len)：提供上下文信息
        # 2. 待预测部分 (pred_len)：初始化为均值或零
        # 
        # 【为什么需要label_len？】
        # 解码器需要一些历史信息作为"桥梁"
        # 连接过去和未来，使预测更平滑
        # ============================================================
        
        # 趋势项输入：取最后label_len部分 + 均值预测
        # [batch, label_len + pred_len, features]
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        
        # 季节项输入：取最后label_len部分 + 零填充
        # 使用pad进行零填充
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        # ============================================================
        # 第三步：编码器处理
        # ============================================================
        # 
        # 【编码器工作流程】
        # 1. 嵌入：将原始数据转换为特征表示
        # 2. 多层处理：每层进行频域注意力 + 分解 + 前馈网络
        # 3. 输出：编码表示，供解码器使用
        # ============================================================
        
        # 嵌入编码器输入
        # enc_out: [batch, seq_len, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 编码器处理
        # enc_out: 编码后的特征
        # attns: 各层的注意力权重（用于可视化）
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # ============================================================
        # 第四步：解码器处理
        # ============================================================
        # 
        # 【解码器工作流程】
        # 1. 嵌入解码器输入（季节项）
        # 2. 自注意力：解码器内部的频域注意力
        # 3. 交叉注意力：查询编码器的信息
        # 4. 前馈网络：进一步处理
        # 5. 每步都进行分解，累积趋势项
        # ============================================================
        
        # 嵌入解码器输入（季节项部分）
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        
        # 解码器处理
        # seasonal_part: 处理后的季节项
        # trend_part: 累积的趋势项
        seasonal_part, trend_part = self.decoder(
            dec_out, 
            enc_out,           # 编码器输出，用于交叉注意力
            x_mask=dec_self_mask, 
            cross_mask=dec_enc_mask,
            trend=trend_init   # 初始趋势项
        )
        
        # ============================================================
        # 第五步：合并输出
        # ============================================================
        # 
        # 【最终预测】
        # 预测 = 趋势项 + 季节项
        # 
        # 这符合时间序列分解的基本原理：
        # 原始序列 = 趋势 + 季节
        # ============================================================
        
        dec_out = trend_part + seasonal_part

        # 返回结果
        if self.output_attention:
            # 返回预测和注意力权重（用于分析）
            return dec_out[:, -self.pred_len:, :], attns
        else:
            # 只返回预测结果
            # 取最后pred_len步，这是预测的未来部分
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, features]


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    # 配置类：定义模型参数
    class Configs(object):
        ab = 0
        modes = 32           # 保留32个频率分量
        mode_select = 'random'  # 随机选择频率
        # version = 'Fourier'    # 傅里叶版本
        version = 'Wavelets'    # 小波版本
        moving_avg = [12, 24]   # 多尺度分解窗口
        L = 1                   # 小波层数
        base = 'legendre'       # 小波基函数
        cross_activation = 'tanh'  # 交叉注意力激活函数
        seq_len = 96            # 输入序列长度
        label_len = 48          # 解码器已知部分长度
        pred_len = 96           # 预测长度
        output_attention = True # 输出注意力权重
        enc_in = 7              # 编码器输入特征数
        dec_in = 7              # 解码器输入特征数
        d_model = 16            # 模型维度
        embed = 'timeF'         # 嵌入类型
        dropout = 0.05          # dropout率
        freq = 'h'              # 时间频率（小时）
        factor = 1
        n_heads = 8             # 注意力头数
        d_ff = 16               # 前馈网络维度
        e_layers = 2            # 编码器层数
        d_layers = 1            # 解码器层数
        c_out = 7               # 输出特征数
        activation = 'gelu'     # 激活函数
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    
    # 创建测试输入
    enc = torch.randn([3, configs.seq_len, 7])       # 编码器输入
    enc_mark = torch.randn([3, configs.seq_len, 4])  # 编码器时间标记
    
    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])  # 解码器输入
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])  # 解码器时间标记
    
    # 运行模型
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)