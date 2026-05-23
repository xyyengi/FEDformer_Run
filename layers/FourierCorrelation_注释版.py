# ============================================================
# 傅里叶相关层 - 详细中文注释版
# ============================================================
# 
# 【本文件核心内容】
# 实现频域注意力机制的核心组件：
# 1. FourierBlock: 傅里叶自注意力块
# 2. FourierCrossAttention: 傅里叶交叉注意力
# 
# 【傅里叶变换基础知识】
# 
# 什么是傅里叶变换？
# -------------------
# 傅里叶变换是将信号从"时域"转换到"频域"的数学工具。
# 
# 时域：信号随时间变化的表现形式
#   - 例如：温度随时间变化的曲线
#   - x轴是时间，y轴是数值
# 
# 频域：信号在不同频率上的分布
#   - 将信号分解为不同频率的正弦波叠加
#   - x轴是频率，y轴是振幅
# 
# 核心思想：
# -------------------
# 任何周期信号都可以表示为不同频率正弦波的加权和：
# 
#   原始信号 = Σ (振幅_k × sin(2π × 频率_k × t + 相位_k))
# 
# 傅里叶变换就是找出这些振幅和相位。
# 
# 为什么在频域做注意力更好？
# -------------------
# 1. 计算效率：复杂度从O(N²)降到O(N log N)
# 2. 物理意义：时间序列的重要信息集中在特定频率
# 3. 全局感受野：频域每个点包含整个序列信息
# 4. 噪声过滤：高频通常是噪声，可以忽略
# 
# ============================================================

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    ============================================================
    频率模式选择函数
    ============================================================
    
    【功能】选择要保留的频率分量索引
    
    【参数说明】
    - seq_len: 序列长度
    - modes: 要保留的频率分量数量
    - mode_select_method: 选择方法
        - 'random': 随机选择频率分量
        - 其他: 选择最低频率（低频通常包含主要信息）
    
    【为什么只选择部分频率？】
    1. 降低计算量：只处理重要的频率
    2. 过滤噪声：高频通常是噪声
    3. 提高泛化：避免过拟合高频细节
    
    【返回值】
    - index: 选中的频率索引列表
    
    【示例】
    seq_len = 96, modes = 32
    - 'random': 随机选择32个频率索引
    - 'lowest': 选择0-31（最低的32个频率）
    ============================================================
    """
    # 确保modes不超过最大可能值（序列长度的一半）
    # 因为rfft只产生 seq_len//2 + 1 个频率分量
    modes = min(modes, seq_len//2)
    
    if mode_select_method == 'random':
        # 随机选择方法
        # 1. 生成所有可能的索引 (0 到 seq_len//2)
        index = list(range(0, seq_len // 2))
        # 2. 随机打乱顺序
        np.random.shuffle(index)
        # 3. 取前modes个
        index = index[:modes]
    else:
        # 选择最低频率方法
        # 低频通常包含趋势和主要周期信息
        index = list(range(0, modes))
    
    # 排序索引（从小到大）
    index.sort()
    return index


# ============================================================
# FourierBlock - 傅里叶自注意力块
# ============================================================
# 
# 【功能】在编码器中实现频域自注意力
# 
# 【自注意力】
# Query = Key = Value（同一个输入）
# 计算输入序列内部的关系
# 
# 【工作流程】
# 1. 输入数据
# 2. 傅里叶变换（时域 → 频域）
# 3. 选择重要频率分量
# 4. 复数乘法（可学习权重）
# 5. 逆傅里叶变换（频域 → 时域）
# 6. 输出结果
# 
# ============================================================

class FourierBlock(nn.Module):
    """
    ============================================================
    傅里叶块 - 频域自注意力实现
    ============================================================
    
    【核心思想】
    将注意力计算从时域转移到频域：
    - 时域注意力：计算所有位置之间的关系，O(N²)
    - 频域注意力：只处理重要频率分量，O(N)
    
    【数学原理】
    1. 傅里叶变换：X(f) = FFT(x(t))
    2. 频域操作：Y(f) = X(f) × W(f)  （复数乘法）
    3. 逆变换：y(t) = IFFT(Y(f))
    
    【复数权重】
    频域表示是复数（包含振幅和相位）
    所以权重也必须是复数，用torch.cfloat类型
    ============================================================
    """
    
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        """
        ============================================================
        初始化傅里叶块
        ============================================================
        
        【参数说明】
        - in_channels: 输入通道数（通常是d_model）
        - out_channels: 输出通道数（通常等于in_channels）
        - seq_len: 序列长度
        - modes: 保留的频率分量数量
        - mode_select_method: 频率选择方法
        ============================================================
        """
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        
        # 选择要保留的频率索引
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        # 缩放因子，用于初始化权重
        # 防止初始权重过大导致训练不稳定
        self.scale = (1 / (in_channels * out_channels))
        
        # ============================================================
        # 可学习的复数权重
        # ============================================================
        # 
        # 【形状解释】
        # [8, in_channels//8, out_channels//8, len(index)]
        # 
        # - 8: 注意力头数（固定）
        # - in_channels//8: 每个头的输入维度
        # - out_channels//8: 每个头的输出维度
        # - len(index): 处理的频率分量数
        # 
        # 【为什么是复数？】
        # 傅里叶变换的结果是复数：
        # - 实部：表示振幅
        # - 虚部：表示相位
        # 
        # 复数乘法可以同时调整振幅和相位
        # ============================================================
        
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                8,                      # 注意力头数
                in_channels // 8,       # 每个头的输入维度
                out_channels // 8,      # 每个头的输出维度
                len(self.index),        # 频率分量数
                dtype=torch.cfloat      # 复数类型！
            )
        )

    def compl_mul1d(self, input, weights):
        """
        ============================================================
        复数乘法 - 频域中的"线性变换"
        ============================================================
        
        【功能】执行复数矩阵乘法
        
        【数学原理】
        复数乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
        
        【einsum解释】
        "bhi,hio->bho"
        - b: batch维度
        - h: 头维度（head）
        - i: 输入维度
        - o: 输出维度
        
        这相当于：output[b,h,o] = Σ input[b,h,i] × weights[h,i,o]
        
        【为什么用einsum？】
        1. 代码简洁
        2. 计算高效
        3. 支持复数运算
        ============================================================
        """
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        """
        ============================================================
        前向传播 - 傅里叶块的核心计算流程
        ============================================================
        
        【完整流程】
        1. 重排维度
        2. 傅里叶变换（时域 → 频域）
        3. 选择性处理频率分量
        4. 复数乘法（应用可学习权重）
        5. 逆傅里叶变换（频域 → 时域）
        
        【输入形状】
        q, k, v: [B, L, H, E]
        - B: batch大小
        - L: 序列长度
        - H: 注意力头数
        - E: 每个头的维度
        
        【注意】虽然接收q, k, v三个参数，但自注意力中它们相同
        ============================================================
        """
        
        # ============================================================
        # 第一步：获取输入形状并重排维度
        # ============================================================
        # 
        # 原始形状: [B, L, H, E]
        # 重排后: [B, H, E, L]
        # 
        # 【为什么重排？】
        # 傅里叶变换需要在最后一个维度上操作
        # 所以把序列长度L放到最后
        # ============================================================
        
        B, L, H, E = q.shape  # 获取形状参数
        x = q.permute(0, 2, 3, 1)  # [B, H, E, L]
        
        # ============================================================
        # 第二步：傅里叶变换
        # ============================================================
        # 
        # torch.fft.rfft: 实数快速傅里叶变换
        # 
        # 【rfft vs fft】
        # - fft: 处理复数输入，输出N个频率分量
        # - rfft: 处理实数输入，输出N//2+1个频率分量
        # 
        # 【为什么用rfft？】
        # 实数信号的傅里叶变换有对称性
        # 只需要一半的频率分量就能完整表示
        # 计算量更小，效率更高
        # 
        # 【输出形状】
        # [B, H, E, L//2+1]，复数类型
        # ============================================================
        
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # ============================================================
        # 第三步：选择性处理频率分量
        # ============================================================
        # 
        # 【核心思想】
        # 不是处理所有频率，只处理选中的重要频率
        # 
        # 【为什么这样做？】
        # 1. 降低计算量
        # 2. 过滤高频噪声
        # 3. 保留重要信息
        # 
        # 【创建输出张量】
        # 初始化为零，只填充选中的频率分量
        # ============================================================
        
        # 创建输出张量（初始化为零）
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        # ============================================================
        # 第四步：复数乘法
        # ============================================================
        # 
        # 对每个选中的频率分量：
        # 1. 取出该频率的傅里叶系数
        # 2. 与可学习权重相乘
        # 3. 存入输出张量
        # 
        # 【物理意义】
        # 权重可以：
        # - 放大/缩小某些频率（调整振幅）
        # - 移动相位（调整时间偏移）
        # - 组合不同通道的信息
        # ============================================================
        
        for wi, i in enumerate(self.index):
            # wi: 权重索引（0, 1, 2, ...）
            # i: 频率索引（选中的频率位置）
            
            # 复数乘法：输入的频率分量 × 对应权重
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        
        # ============================================================
        # 第五步：逆傅里叶变换
        # ============================================================
        # 
        # torch.fft.irfft: 逆实数快速傅里叶变换
        # 
        # 【功能】将频域表示转换回时域
        # 
        # 【参数n】
        # 指定输出长度，确保与原始输入长度相同
        # 
        # 【输出】
        # 实数张量，形状 [B, H, E, L]
        # ============================================================
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        
        # 返回结果（第二个返回值None是占位符，用于兼容接口）
        return (x, None)


# ============================================================
# FourierCrossAttention - 傅里叶交叉注意力
# ============================================================
# 
# 【功能】在解码器中实现频域交叉注意力
# 
# 【交叉注意力 vs 自注意力】
# - 自注意力：Q = K = V（同一个输入）
# - 交叉注意力：Q来自一个输入，K和V来自另一个输入
# 
# 【在FEDformer中的应用】
# - Query: 来自解码器（待预测部分）
# - Key/Value: 来自编码器（历史信息的编码）
# 
# 【作用】
# 让解码器"查询"编码器中的历史信息
# 找出历史数据中与当前预测相关的部分
# 
# ============================================================

class FourierCrossAttention(nn.Module):
    """
    ============================================================
    傅里叶交叉注意力 - 频域中的Query-Key-Value机制
    ============================================================
    
    【核心思想】
    在频域中实现交叉注意力：
    1. Query和解码器输入做傅里叶变换
    2. Key和编码器输出做傅里叶变换
    3. 在频域计算注意力权重
    4. 加权求和Value
    5. 逆变换回时域
    
    【数学原理】
    频域注意力：Attention(Q, K, V) = softmax(Q·Kᵀ) × V
    在频域中，这个计算更高效
    ============================================================
    """
    
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        """
        ============================================================
        初始化傅里叶交叉注意力
        ============================================================
        
        【参数说明】
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - seq_len_q: Query序列长度（解码器）
        - seq_len_kv: Key/Value序列长度（编码器）
        - modes: 保留的频率分量数量
        - mode_select_method: 频率选择方法
        - activation: 激活函数类型
            - 'tanh': 使用tanh激活（适合复数）
            - 'softmax': 使用softmax（需要取绝对值）
        ============================================================
        """
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 分别为Query和Key/Value选择频率索引
        # 因为它们的序列长度可能不同
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        
        # 可学习的复数权重
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                8, 
                in_channels // 8, 
                out_channels // 8, 
                len(self.index_q), 
                dtype=torch.cfloat
            )
        )

    def compl_mul1d(self, input, weights):
        """
        复数乘法（同FourierBlock）
        """
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        """
        ============================================================
        前向传播 - 傅里叶交叉注意力的核心计算
        ============================================================
        
        【完整流程】
        1. 重排维度
        2. Query傅里叶变换
        3. Key傅里叶变换
        4. 频域注意力计算（Q·Kᵀ）
        5. 激活函数
        6. 加权求和Value
        7. 应用可学习权重
        8. 逆傅里叶变换
        
        【输入】
        - q: Query，来自解码器 [B, L_q, H, E]
        - k: Key，来自编码器 [B, L_k, H, E]
        - v: Value，来自编码器 [B, L_k, H, E]
        ============================================================
        """
        
        # ============================================================
        # 第一步：获取形状并重排维度
        # ============================================================
        B, L, H, E = q.shape
        
        # 重排维度：把序列长度放到最后
        xq = q.permute(0, 2, 3, 1)   # [B, H, E, L_q]
        xk = k.permute(0, 2, 3, 1)   # [B, H, E, L_k]
        xv = v.permute(0, 2, 3, 1)   # [B, H, E, L_k]

        # ============================================================
        # 第二步：Query的傅里叶变换
        # ============================================================
        # 
        # 创建存储张量，只保留选中的频率分量
        # ============================================================
        
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        
        # 傅里叶变换
        xq_ft = torch.fft.rfft(xq, dim=-1)
        
        # 只保留选中的频率分量
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        
        # ============================================================
        # 第三步：Key的傅里叶变换
        # ============================================================
        # 同样的处理，但使用Key的频率索引
        # ============================================================
        
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        
        # 傅里叶变换
        xk_ft = torch.fft.rfft(xk, dim=-1)
        
        # 只保留选中的频率分量
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # ============================================================
        # 第四步：频域注意力计算
        # ============================================================
        # 
        # 【核心计算】Q·Kᵀ
        # 
        # einsum "bhex,bhey->bhxy":
        # - b: batch
        # - h: head
        # - e: embedding维度
        # - x: Query的频率索引
        # - y: Key的频率索引
        # 
        # 结果：注意力权重矩阵 [B, H, x, y]
        # ============================================================
        
        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
        
        # ============================================================
        # 第五步：激活函数
        # ============================================================
        # 
        # 【为什么需要激活函数？】
        # 将注意力权重归一化，使其在合理范围内
        # 
        # 【tanh vs softmax】
        # - tanh: 直接对复数应用，输出范围[-1, 1]
        # - softmax: 需要先取绝对值，再归一化
        # 
        # 【为什么用tanh？】
        # 1. 可以处理复数
        # 2. 输出有正有负，保留更多信息
        # ============================================================
        
        if self.activation == 'tanh':
            # tanh激活：直接对复数应用
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            # softmax激活：先取绝对值，再归一化
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            # 转回复数（虚部为0）
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        
        # ============================================================
        # 第六步：加权求和Value
        # ============================================================
        # 
        # 【计算】注意力权重 × Key的频域表示
        # 
        # einsum "bhxy,bhey->bhex":
        # - 用注意力权重加权Key的频域信息
        # - 结果形状：[B, H, E, x]（Query的频率维度）
        # ============================================================
        
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        
        # ============================================================
        # 第七步：应用可学习权重
        # ============================================================
        # 
        # 【计算】加权结果 × 可学习权重
        # 
        # einsum "bhex,heox->bhox":
        # - 进一步变换，引入可学习参数
        # ============================================================
        
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        
        # ============================================================
        # 第八步：逆傅里叶变换
        # ============================================================
        # 
        # 1. 将结果放回完整的频率张量
        # 2. 逆变换回时域
        # ============================================================
        
        # 创建完整频率张量
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        
        # 将处理后的频率分量放回对应位置
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        
        # 逆傅里叶变换
        # 注意：除以in_channels和out_channels进行归一化
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        
        return (out, None)


# ============================================================
# 使用示例和测试
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("傅里叶相关层测试")
    print("=" * 60)
    
    # 测试参数
    batch_size = 3
    seq_len = 96
    d_model = 16
    n_heads = 8
    d_head = d_model // n_heads  # 每个头的维度
    modes = 32
    
    # 创建测试输入
    # 自注意力测试
    x = torch.randn(batch_size, seq_len, n_heads, d_head)
    
    print("\n--- FourierBlock 测试 ---")
    fourier_block = FourierBlock(
        in_channels=d_model,
        out_channels=d_model,
        seq_len=seq_len,
        modes=modes,
        mode_select_method='random'
    )
    
    output, _ = fourier_block(x, x, x, None)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 交叉注意力测试
    seq_len_q = 48 + 96  # label_len + pred_len
    seq_len_kv = 96
    
    q = torch.randn(batch_size, seq_len_q, n_heads, d_head)
    k = torch.randn(batch_size, seq_len_kv, n_heads, d_head)
    v = torch.randn(batch_size, seq_len_kv, n_heads, d_head)
    
    print("\n--- FourierCrossAttention 测试 ---")
    fourier_cross = FourierCrossAttention(
        in_channels=d_model,
        out_channels=d_model,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        modes=modes,
        mode_select_method='random',
        activation='tanh'
    )
    
    output, _ = fourier_cross(q, k, v, None)
    print(f"Query形状: {q.shape}")
    print(f"Key形状: {k.shape}")
    print(f"输出形状: {output.shape}")
    
    print("\n测试完成！")