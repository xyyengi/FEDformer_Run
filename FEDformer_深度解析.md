# FEDformer 深度解析 - 从零开始理解频域注意力机制

> 本文档面向没有深度学习基础的读者，详细解释FEDformer的每一个组件和概念。

---

## 目录
1. [什么是时间序列预测？](#1-什么是时间序列预测)
2. [Transformer基础回顾](#2-transformer基础回顾)
3. [FEDformer的核心创新](#3-fedformer的核心创新)
4. [数据流完整解析](#4-数据流完整解析)
5. [频域注意力机制详解](#5-频域注意力机制详解)
6. [分解机制详解](#6-分解机制详解)
7. [为什么FEDformer更好？](#7-为什么fedformer更好)
8. [代码与概念对照表](#8-代码与概念对照表)

---

## 1. 什么是时间序列预测？

### 1.1 基本概念

**时间序列**：按时间顺序排列的数据点序列。
- 例如：每天的气温、每小时的电力负荷、每分钟的股票价格
- 你的项目中的数据：Wind（风力）、Solar（太阳能）、Load（负荷）

**时间序列预测**：根据历史数据预测未来值。
- 输入：过去336小时的数据（seq_len = 336）
- 输出：未来168小时的预测（pred_len = 168）

### 1.2 时间序列的两个组成部分

任何时间序列都可以分解为：

```
原始序列 = 趋势项 (Trend) + 季节项 (Seasonal)
```

- **趋势项**：长期变化方向（如逐年上升的温度）
- **季节项**：周期性波动（如每天的温度变化、每周的消费模式）

**类比理解**：
- 趋势项就像"大势"——比如经济整体增长
- 季节项就像"波动"——比如节假日消费高峰

---

## 2. Transformer基础回顾

### 2.1 Transformer是什么？

Transformer是2017年提出的深度学习模型，最初用于机器翻译，后来广泛应用于各种任务。

**核心思想**：注意力机制（Attention）

```
注意力 = 查询(Query) × 键(Key)ᵀ × 值(Value)
```

**通俗理解**：
- Query：我想找什么
- Key：每个位置的标签
- Value：每个位置的实际内容
- 注意力：计算Query和每个Key的相似度，然后加权求和Value

### 2.2 传统Transformer的问题

**问题1：计算复杂度高**
- 传统注意力需要计算所有位置之间的关系
- 复杂度：O(N²)，N是序列长度
- 序列越长，计算量爆炸式增长

**问题2：对长序列建模困难**
- 长距离依赖难以捕捉
- 信息在传递过程中逐渐丢失

**问题3：对周期性模式不敏感**
- 时间序列通常有周期性（日周期、周周期、年周期）
- 传统Transformer无法有效利用这种周期性

---

## 3. FEDformer的核心创新

### 3.1 两大创新点

**创新1：频域注意力（Frequency Enhanced Attention）**
- 将时间序列转换到频域
- 在频域中进行注意力计算
- 复杂度从O(N²)降到O(N log N)或O(N)

**创新2：分解架构（Decomposition Architecture）**
- 将序列分解为趋势项和季节项
- 分别处理，最后合并
- 更好地捕捉不同时间尺度的模式

### 3.2 两种频域方法

FEDformer提供两种版本：

| 版本 | 方法 | 特点 |
|------|------|------|
| Fourier | 傅里叶变换 | 全局频率分析，适合捕捉整体周期 |
| Wavelets | 小波变换 | 多尺度分析，同时捕捉全局和局部模式 |

---

## 4. 数据流完整解析

### 4.1 整体流程图

```
输入数据 (历史序列)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第一步：数据预处理                                        │
│  • 归一化（MinMaxScaler）                                │
│  • 时间特征编码（sin/cos周期编码）                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第二步：序列分解                                         │
│  • 移动平均提取趋势项                                     │
│  • 原始序列 - 趋势项 = 季节项                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第三步：嵌入层（Embedding）                              │
│  • 值嵌入（TokenEmbedding）：1D卷积提取特征               │
│  • 时间嵌入（TemporalEmbedding）：编码时间信息            │
│  • 注意：FEDformer不使用位置嵌入！                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第四步：编码器（Encoder）                                │
│  • 多层EncoderLayer                                       │
│  • 每层：频域自注意力 → 分解 → 前馈网络 → 分解             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第五步：解码器（Decoder）                                │
│  • 自注意力（频域）→ 分解                                 │
│  • 交叉注意力（频域，与编码器输出交互）→ 分解               │
│  • 前馈网络 → 分解                                        │
│  • 累加所有趋势项                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  第六步：输出                                             │
│  • 季节项 + 趋势项 = 最终预测                             │
│  • 反归一化                                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
预测结果 (未来序列)
```

### 4.2 数据形状变化

以你的项目配置为例：
- seq_len = 336（输入序列长度）
- label_len = 168（解码器输入的已知部分）
- pred_len = 168（预测长度）
- d_model = 512（模型维度）
- n_heads = 8（注意力头数）

```
输入数据形状变化：

原始数据: [batch_size, 336, 3]  # 3个特征：Wind, Solar, Load
    │
    ▼ 嵌入层
编码器输入: [batch_size, 336, 512]  # 3 → 512 维度扩展
    │
    ▼ 编码器
编码器输出: [batch_size, 336, 512]
    │
    ▼ 解码器
解码器输入: [batch_size, 336, 512]  # label_len(168) + pred_len(168) = 336
    │
    ▼ 解码器
预测输出: [batch_size, 168, 3]  # 只取最后168步，维度还原
```

---

## 5. 频域注意力机制详解

### 5.1 什么是频域？

**时域 vs 频域**：

- **时域**：我们通常看到的数据形式，x轴是时间，y轴是数值
- **频域**：将数据表示为不同频率的正弦波的叠加，x轴是频率，y轴是振幅

**类比理解**：
- 时域就像听到的音乐（随时间变化的声音）
- 频域就像音乐的乐谱（不同频率的音符组合）

### 5.2 傅里叶变换（Fourier Transform）

**核心思想**：任何周期函数都可以表示为不同频率正弦波的叠加

```
时域信号 ──傅里叶变换──▶ 频域表示
频域表示 ──逆傅里叶变换──▶ 时域信号
```

**数学公式**：
```
X(f) = ∫ x(t) * e^(-2πift) dt

其中：
- x(t) 是时域信号
- X(f) 是频域表示
- e^(-2πift) 是复数正弦波
```

**代码对应**（layers/FourierCorrelation.py）：
```python
# 傅里叶变换
x_ft = torch.fft.rfft(x, dim=-1)  # 时域 → 频域

# 逆傅里叶变换
x = torch.fft.irfft(out_ft, n=x.size(-1))  # 频域 → 时域
```

### 5.3 为什么在频域做注意力更好？

**原因1：计算效率**
- 时域注意力：O(N²)
- 频域注意力：O(N log N) 或 O(N)（只保留部分频率）

**原因2：物理意义**
- 时间序列的重要模式通常集中在低频（趋势）和特定频率（周期）
- 高频通常是噪声，可以忽略
- 通过选择性地保留频率分量，可以：
  - 降低计算量
  - 提高模型泛化能力
  - 更好地捕捉周期性模式

**原因3：全局感受野**
- 频域中每个点都包含整个序列的信息
- 天然具有全局视野，不需要像Transformer那样逐层扩大感受野

### 5.4 FourierBlock详解

**位置**：layers/FourierCorrelation.py

**功能**：编码器中的自注意力

**工作流程**：

```
输入 x: [B, L, H, E]
    │
    ▼
1. 重排维度: [B, H, E, L]
    │
    ▼
2. 傅里叶变换: rfft → [B, H, E, L//2+1] (复数)
    │
    ▼
3. 选择频率模式: 只保留重要的频率分量
    │
    ▼
4. 复数乘法: 与可学习权重相乘
    │
    ▼
5. 逆傅里叶变换: irfft → [B, H, E, L]
    │
    ▼
输出: [B, L, H, E]
```

**关键代码解析**：

```python
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        # modes: 保留的频率分量数量
        # mode_select_method: 频率选择方法
        #   - 'random': 随机选择
        #   - 'lowest': 选择最低频率（通常包含主要信息）
        
        # 选择要保留的频率索引
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        
        # 可学习的复数权重
        self.weights1 = nn.Parameter(
            torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat)
        )
    
    def forward(self, q, k, v, mask):
        # 1. 傅里叶变换
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # 2. 只处理选中的频率分量
        out_ft = torch.zeros(B, H, E, L // 2 + 1, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            # 复数乘法：输入 × 权重
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        
        # 3. 逆傅里叶变换
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
```

### 5.5 FourierCrossAttention详解

**位置**：layers/FourierCorrelation.py

**功能**：解码器中的交叉注意力（Query来自解码器，Key/Value来自编码器）

**与自注意力的区别**：
- 自注意力：Q = K = V（同一个输入）
- 交叉注意力：Q来自一个输入，K和V来自另一个输入

**工作流程**：

```
Query (解码器): [B, L_q, H, E]
Key (编码器): [B, L_k, H, E]
Value (编码器): [B, L_k, H, E]
    │
    ▼
1. 分别进行傅里叶变换
    │
    ▼
2. 在频域计算注意力
   Q·Kᵀ → 注意力权重
    │
    ▼
3. 加权求和Value
    │
    ▼
4. 逆傅里叶变换
    │
    ▼
输出: [B, L_q, H, E]
```

**关键代码解析**：

```python
def forward(self, q, k, v, mask):
    # 1. Query的傅里叶变换
    xq_ft = torch.fft.rfft(xq, dim=-1)
    xq_ft_ = 只保留选中的频率分量
    
    # 2. Key的傅里叶变换
    xk_ft = torch.fft.rfft(xk, dim=-1)
    xk_ft_ = 只保留选中的频率分量
    
    # 3. 频域注意力计算
    # Q·Kᵀ (在频域)
    xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_)
    
    # 激活函数（tanh或softmax）
    xqk_ft = xqk_ft.tanh()
    
    # 4. 加权求和Value
    # 注意力权重 × V
    xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
    
    # 5. 应用可学习权重
    xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
    
    # 6. 逆傅里叶变换
    out = torch.fft.irfft(out_ft, n=xq.size(-1))
    return out
```

### 5.6 小波变换（Wavelet Transform）

**位置**：layers/MultiWaveletCorrelation.py

**与傅里叶变换的区别**：

| 特性 | 傅里叶变换 | 小波变换 |
|------|-----------|---------|
| 频率分辨率 | 全局统一 | 多尺度，可变 |
| 时间信息 | 丢失 | 保留 |
| 适用场景 | 平稳信号 | 非平稳信号 |
| 计算复杂度 | O(N log N) | O(N) |

**为什么需要小波？**

傅里叶变换的问题：
- 只能告诉我们"有哪些频率"
- 不能告诉我们"这些频率在什么时候出现"

小波变换的优势：
- 可以同时分析"什么频率"和"什么时候"
- 适合分析有突变或局部特征的时间序列

**小波分解示意**：

```
原始信号
    │
    ▼ 小波分解
┌─────────────┬─────────────┐
│  低频部分    │  高频部分    │
│  (近似系数)  │  (细节系数)  │
│  代表趋势    │  代表细节    │
└─────────────┴─────────────┘
    │
    ▼ 继续分解低频部分
┌─────────────┬─────────────┐
│  更低频     │  中频       │
└─────────────┴─────────────┘
```

**代码对应**：

```python
def wavelet_transform(self, x):
    # 分解为低频(近似)和高频(细节)
    xa = torch.cat([x[:, ::2, :, :],  # 偶数位置
                    x[:, 1::2, :, :],  # 奇数位置
                    ], -1)
    d = torch.matmul(xa, self.ec_d)  # 高频(细节)
    s = torch.matmul(xa, self.ec_s)  # 低频(近似)
    return d, s

def evenOdd(self, x):
    # 重构：从低频和高频恢复原始信号
    x_e = torch.matmul(x, self.rc_e)  # 偶数位置
    x_o = torch.matmul(x, self.rc_o)  # 奇数位置
    # 交错排列
    x[..., ::2, :, :] = x_e
    x[..., 1::2, :, :] = x_o
    return x
```

---

## 6. 分解机制详解

### 6.1 什么是序列分解？

**核心思想**：将时间序列分解为趋势项和季节项

```
原始序列 = 趋势项 + 季节项
```

**为什么需要分解？**
- 趋势项和季节项有不同的特性
- 分别处理可以更好地建模
- 最后合并得到更准确的预测

### 6.2 移动平均分解

**位置**：layers/Autoformer_EncDec.py

**原理**：使用移动平均提取趋势

```python
class moving_avg(nn.Module):
    """
    移动平均：平滑数据，提取趋势
    """
    def __init__(self, kernel_size, stride):
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        # 边界填充
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # 移动平均
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    序列分解
    """
    def __init__(self, kernel_size):
        self.moving_avg = moving_avg(kernel_size, stride=1)
    
    def forward(self, x):
        # 趋势项 = 移动平均
        moving_mean = self.moving_avg(x)
        # 季节项 = 原始 - 趋势
        res = x - moving_mean
        return res, moving_mean  # 返回季节项和趋势项
```

**图示**：

```
原始序列:  /\_/\_/\_/\_/\_/
              │
              ▼ 移动平均
趋势项:   __________________  (平滑的线)
              │
              ▼ 原始 - 趋势
季节项:   /\_/\_/\_/\_/\_/  (周期性波动)
```

### 6.3 渐进式分解架构

**位置**：编码器层和解码器层

**核心思想**：在每一层都进行分解，逐步精炼趋势和季节

**编码器层**：

```python
class EncoderLayer(nn.Module):
    def forward(self, x, attn_mask=None):
        # 1. 频域自注意力
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        # 2. 第一次分解
        x, _ = self.decomp1(x)
        
        # 3. 前馈网络
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        
        # 4. 第二次分解
        res, _ = self.decomp2(x + y)
        return res, attn
```

**解码器层**：

```python
class DecoderLayer(nn.Module):
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. 自注意力 + 分解
        x = x + self.dropout(self.self_attention(x, x, x)[0])
        x, trend1 = self.decomp1(x)
        
        # 2. 交叉注意力 + 分解
        x = x + self.dropout(self.cross_attention(x, cross, cross)[0])
        x, trend2 = self.decomp2(x)
        
        # 3. 前馈网络 + 分解
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        x, trend3 = self.decomp3(x + y)
        
        # 4. 累加所有趋势项
        residual_trend = trend1 + trend2 + trend3
        return x, residual_trend
```

### 6.4 为什么渐进式分解有效？

**原因1：逐步精炼**
- 每一层都提取趋势，逐步去除噪声
- 季节项在层间传递，保留周期性信息

**原因2：信息流动**
- 趋势信息通过专门的路径累积
- 季节信息通过注意力机制增强

**原因3：稳定性**
- 分解使训练更稳定
- 避免梯度消失/爆炸

---

## 7. 为什么FEDformer更好？

### 7.1 与Transformer对比

| 方面 | Transformer | FEDformer |
|------|-------------|-----------|
| 注意力计算 | 时域，O(N²) | 频域，O(N)或O(N log N) |
| 长序列处理 | 困难，计算量大 | 高效，只处理重要频率 |
| 周期性建模 | 隐式学习 | 显式利用频域特性 |
| 趋势建模 | 无专门机制 | 分解架构专门处理 |
| 可解释性 | 较差 | 频域权重可解释 |

### 7.2 与其他时序模型对比

| 模型 | 特点 | 局限性 |
|------|------|--------|
| RNN/LSTM | 序列建模 | 长距离依赖困难 |
| Transformer | 全局注意力 | O(N²)复杂度 |
| Informer | 稀疏注意力 | 仍有时域限制 |
| Autoformer | 自相关+分解 | 时域自相关 |
| **FEDformer** | **频域注意力+分解** | **高效+有效** |

### 7.3 理论优势

**1. 频域稀疏性**
- 时间序列的重要信息通常集中在少数频率
- 通过选择性地处理频率分量，大幅降低计算量

**2. 全局感受野**
- 频域中每个点都包含整个序列的信息
- 天然具有全局视野

**3. 周期性建模**
- 频域天然适合表示周期性
- 傅里叶变换的基函数就是正弦波

**4. 噪声过滤**
- 高频通常是噪声
- 通过忽略高频分量，自动去噪

### 7.4 实验结果

根据原论文，FEDformer在多个数据集上取得了SOTA（State-of-the-Art）结果：

- 在长序列预测任务上，比Transformer平均提升约30%
- 计算效率提升约80%
- 在多变量预测任务上表现尤为突出

---

## 8. 代码与概念对照表

### 8.1 核心文件结构

```
FEDformer_Run/
├── models/
│   └── FEDformer.py          # 主模型定义
├── layers/
│   ├── FourierCorrelation.py  # 傅里叶注意力
│   ├── MultiWaveletCorrelation.py  # 小波注意力
│   ├── Autoformer_EncDec.py   # 编码器/解码器 + 分解
│   └── Embed.py               # 嵌入层
├── exp/
│   └── exp_main.py            # 训练/测试流程
└── data_provider/
    └── data_loader.py         # 数据加载
```

### 8.2 关键类与功能对照

| 类名 | 文件 | 功能 |
|------|------|------|
| `Model` | FEDformer.py | 主模型，整合所有组件 |
| `FourierBlock` | FourierCorrelation.py | 傅里叶自注意力 |
| `FourierCrossAttention` | FourierCorrelation.py | 傅里叶交叉注意力 |
| `MultiWaveletTransform` | MultiWaveletCorrelation.py | 小波自注意力 |
| `MultiWaveletCross` | MultiWaveletCorrelation.py | 小波交叉注意力 |
| `series_decomp` | Autoformer_EncDec.py | 序列分解 |
| `moving_avg` | Autoformer_EncDec.py | 移动平均 |
| `Encoder` | Autoformer_EncDec.py | 编码器 |
| `Decoder` | Autoformer_EncDec.py | 解码器 |
| `EncoderLayer` | Autoformer_EncDec.py | 编码器层 |
| `DecoderLayer` | Autoformer_EncDec.py | 解码器层 |
| `DataEmbedding_wo_pos` | Embed.py | 数据嵌入（无位置编码） |
| `TokenEmbedding` | Embed.py | 值嵌入 |
| `TemporalEmbedding` | Embed.py | 时间嵌入 |

### 8.3 数据流关键代码

**1. 模型初始化**（models/FEDformer.py）

```python
class Model(nn.Module):
    def __init__(self, configs):
        # 选择频域方法
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(...)  # 小波
        else:
            encoder_self_att = FourierBlock(...)  # 傅里叶
        
        # 分解模块
        self.decomp = series_decomp(kernel_size)
        
        # 嵌入层（无位置编码）
        self.enc_embedding = DataEmbedding_wo_pos(...)
        self.dec_embedding = DataEmbedding_wo_pos(...)
        
        # 编码器和解码器
        self.encoder = Encoder([...])
        self.decoder = Decoder([...])
```

**2. 前向传播**（models/FEDformer.py）

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, ...):
    # 1. 初始分解
    seasonal_init, trend_init = self.decomp(x_enc)
    
    # 2. 准备解码器输入
    trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
    seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], ...)
    
    # 3. 编码器
    enc_out = self.enc_embedding(x_enc, x_mark_enc)
    enc_out, attns = self.encoder(enc_out)
    
    # 4. 解码器
    dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
    seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
    
    # 5. 合并输出
    dec_out = trend_part + seasonal_part
    return dec_out[:, -self.pred_len:, :]
```

**3. 傅里叶注意力**（layers/FourierCorrelation.py）

```python
def forward(self, q, k, v, mask):
    # 时域 → 频域
    x_ft = torch.fft.rfft(x, dim=-1)
    
    # 选择性处理频率分量
    out_ft = torch.zeros(...)
    for wi, i in enumerate(self.index):
        out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
    
    # 频域 → 时域
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x
```

**4. 序列分解**（layers/Autoformer_EncDec.py）

```python
def forward(self, x):
    # 趋势 = 移动平均
    moving_mean = self.moving_avg(x)
    # 季节 = 原始 - 趋势
    res = x - moving_mean
    return res, moving_mean
```

---

## 附录：常见问题解答

### Q1: 为什么FEDformer不使用位置编码？

**A**: 频域注意力天然具有全局感受野，每个频率分量都包含整个序列的信息，因此不需要额外的位置编码来表示位置信息。

### Q2: modes参数是什么意思？

**A**: modes表示保留的频率分量数量。较小的modes会：
- 降低计算量
- 过滤更多高频噪声
- 但可能丢失有用信息

### Q3: 如何选择Fourier还是Wavelets？

**A**: 
- **Fourier**: 适合有明显周期性的数据，计算更快
- **Wavelets**: 适合有突变或多尺度模式的数据，更灵活

### Q4: 分解的kernel_size如何选择？

**A**: kernel_size决定了趋势的平滑程度：
- 较大值：更平滑的趋势，适合长周期
- 较小值：保留更多细节，适合短周期
- 可以使用多个kernel_size的组合

### Q5: 为什么解码器输入需要label_len？

**A**: 解码器需要一些历史信息作为"上下文"：
- label_len: 已知的历史部分
- pred_len: 需要预测的未来部分（初始化为0或均值）

---

## 总结

FEDformer通过两个关键创新解决了时间序列预测的挑战：

1. **频域注意力**：将计算从时域转移到频域，利用频域的稀疏性和全局性，大幅降低计算复杂度，同时更好地捕捉周期性模式。

2. **分解架构**：将序列分解为趋势和季节项，分别处理后合并，使模型能够更好地建模不同时间尺度的模式。

这两个创新结合，使FEDformer在长序列时间序列预测任务上取得了优异的性能。

---

*文档创建时间：2024年*
*适用于FEDformer代码库*