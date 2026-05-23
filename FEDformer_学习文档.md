# FEDformer 项目学习与复现指南

本文档旨在帮助你快速理解 FEDformer 的项目架构、数据处理逻辑，并记录了如何针对自定义的风光负荷数据集进行重构与 168 小时（一周）长序列预测的完整流程。

## 一、 环境配置总结

FEDformer 项目依赖较早版本的 PyTorch 和 Pandas。在当前环境中，我们使用了 **Python 3.8** 虚拟环境进行隔离配置。主要步骤如下：
1. 创建并激活 Python 3.8 虚拟环境。
2. 安装 `requirements.txt` 中的依赖（其中修正了 `pytorch-wavelet` 的官方 GitHub 源码安装路径：`git+https://github.com/fbcotter/pytorch_wavelets`）。
3. 安装了 `pandas==1.4.2` 等兼容的数据处理包，确保底层 C 扩展正常编译。

---

## 二、 FEDformer 项目框架分析

整个项目是一个标准的基于 PyTorch 的时间序列深度学习项目，各模块分工明确。下面是核心目录结构图解：

```text
FEDformer/
├── run.py                 --- 项目的总入口文件，负责解析命令行参数并启动实验
├── data_provider/         --- [数据模块] 负责时间序列数据的加载和预处理
│   ├── data_factory.py    --- 数据集构建工厂，生成对应的 DataLoader
│   └── data_loader.py     --- 自定义 PyTorch Dataset，处理时间特征编码 (Time Feature Encoding) 等
├── models/                --- [模型模块] 存放所有的网络主结构
│   ├── FEDformer.py       --- 本文提出的核心模型
│   ├── Autoformer.py      --- 也是一种时序经典模型，作为 Baseline 或组件
│   └── ... 
├── layers/                --- [层级模块] 组成上述模型的算子和网络层
│   ├── FourierCorrelation.py      --- 傅里叶变换相关性提取 (FEA 核心)
│   ├── MultiWaveletCorrelation.py --- 多小波变换相关性提取
│   ├── Transformer_EncDec.py      --- 经典的 Transformer 编码器-解码器架构抽象
│   └── SelfAttention_Family.py    --- 各种自注意力的实现机制
├── exp/                   --- [实验/训练模块] 负责前向传播外的训练、验证、测试流程控制
│   ├── exp_basic.py       --- 实验环境基础类 (比如设备控制 GPU/CPU)
│   └── exp_main.py        --- 核心训练与测试循环 (包含 Epoch 迭代、Loss 计算、反向传播更新)
├── utils/                 --- [工具模块] 通用工具代码
│   ├── metrics.py         --- 预测评估指标分析：MAE, MSE, RMSE, MAPE, MSPE 等
│   └── timefeatures.py    --- 将时间戳转换为数值特征（如周几，月，日等）
└── scripts/               --- [执行模块] 作者打包好的各种数据集下的最优超参数脚本
    ├── run_M.sh           --- Multivariate (多变量输入 -> 多变量预测)
    └── run_S.sh           --- Univariate (单变量输入 -> 单变量预测)
```

### 核心机制导读
1. **入口解析**：`run.py` 使用 `argparse` 定义了预测长度 (`pred_len`)、输入长度 (`seq_len`)、以及特征类型 (`features: M, S, MS`) 等关键参数。
2. **频域注意力机制 (FEA/FEB)**：`layers/FourierCorrelation.py` & `layers/MultiWaveletCorrelation.py`。FEDformer 的核心创新点，利用 `torch.fft` 将时间序列投射到频域来计算注意力机制，把复杂度从 $O(L^2)$ 降到 $O(L)$。
3. **分解机制 (Decomposition)**：复用 `layers/Autoformer_EncDec.py` 中的系列分解层，将时间序列分解为趋势项（Trend）和季节项（Seasonal），有利于长期预测。

---

## 三、 数据加载与预处理逻辑 (`data_loader.py`)

如果你要换用自己的数据集，`data_provider/data_loader.py` 中的逻辑如下：

1. **数据格式要求**：默认期望标准 CSV 文件。第一列必须是时间戳/日期（默认列名 `date`）。后续列为数值特征。通过 `OT` 列指定目标预测列（单变量预测时）。
2. **数据集切分**：如果在 `run.py` 传入 `--data custom`，系统会调用 `Dataset_Custom` 类，按 `0.7 : 0.1 : 0.2` 的比例自动将数据切分为训练集、验证集和测试集。
3. **数据标准化 (Standardization)**：系统内部会在 `scale=True`（默认）时执行标准化操作（$0$ 均值，$1$ 方差）。**注意**：程序严格只使用**训练集**的数据来 `fit` 计算均值和方差，防止引入“未来数据”。预测输出会自动逆变换回原量纲。
4. **时间特征编码**：通过 `--freq h` 识别数据频率，内部调用 `utils/timefeatures.py`，将提取的时、日、月等时间特征编码为 $[-1, 1]$ 之间的高频正弦/余弦特征（设置 `timeenc=1`），帮助模型学习周期性。
5. **滑动窗口切片**：`__getitem__` 根据 `seq_len`（过去输入步长）、`label_len`（Decoder初始带入的已知历史标签步长）和 `pred_len`（预测未来步长）来切取数据块。

---

## 四、 德国公开风光负荷数据集的重构过程

我们有两个原始公开数据集：`Actual_consumption` (用电负荷) 和 `Actual_generation` (发电数据)。由于存在分号分隔、带千位符的字符串数值格式以及多余列，我们必须进行重构，代码逻辑存在于 `preprocess_data.py` 中。

**处理逻辑如下：**
1. **合并数据表**：通过 pandas 根据相同的 `Start date` 取交集 (`inner join`)。
2. **清洗数值格式**：编写 `clean_numeric` 函数，去掉如 `38,346.00` 中的逗号 `,`，并强制转换为浮点型 `float`，防止模型读取为字符串报错。
3. **抽取特征并聚合**：
   - 提取 `grid load [MWh] Calculated resolutions` 列重命名为 `Load`。
   - 提取 `Photovoltaics [MWh] Calculated resolutions` 列重命名为 `Solar`。
   - 提取 `Wind offshore` (海上风电) 与 `Wind onshore` (陆上风电)，将缺失值填充为 $0$ 后**两列相加**，统一合并为 `Wind` (总风电) 变量。
4. **统一时间戳格式**：将字符串时间转换为 pandas 的 `datetime`，舍弃原有的冗余时间列，重命名为 `date` 列。
5. **清洗并保存**：执行 `dropna()` 丢弃缺失行，并按时间重新排序，最终保存出标准 CSV 文件 **`Wind_Solar_Load.csv`**。

重构后的数据集包含 4 列 `[date, Wind, Solar, Load]`，共 27743 行，十分干净。

---

## 五、 如何使用新数据集预测 168 小时（一周）

我们的目标是多变量输入预测多变量输出（Wind, Solar, Load 共同输入，共同预测未来 168 个小时的值）。特征数为 3，`features='M'`。

在虚拟环境激活状态下（`source venv/bin/activate`），运行以下命令即可开始训练与预测：

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./ \
  --data_path Wind_Solar_Load.csv \
  --task_id wind_solar_load_168 \
  --model FEDformer \
  --data custom \
  --features M \
  --seq_len 336 \
  --label_len 168 \
  --pred_len 168 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --freq h \
  --des 'Exp_Custom' \
  --itr 1
```

**参数映射解析：**
* `--data custom`：使用 `Dataset_Custom` 加载器。
* `--data_path Wind_Solar_Load.csv`：读取我们刚生成的清洗数据。
* `--seq_len 336`：输入过去 14 天（336小时）的历史特征。
* `--pred_len 168`：精准控制模型输出预测未来 7 天（168小时）。
* `--enc_in 3, --dec_in 3, --c_out 3`：表示输入和输出的变量维度为 3（风、光、负荷）。