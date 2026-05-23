# FEDformer 风光荷 168h 预测项目

本项目基于 FEDformer，用于德国风电、光伏、负荷的多变量长序列预测。当前主线已经完成数据时间轴修复、严格时间切分、Dataset/DataLoader smoke test 和 CPU 小规模 168 -> 168 验证。

当前推荐任务：

- 输入窗口：`168` 小时
- decoder label history：`168` 小时
- 预测窗口：`168` 小时
- 特征：`Wind / Solar / Load + 8 维周期时间特征`
- residual 约定：`residual = actual - forecast`

## 当前数据版本

主数据文件：

```text
Wind_Solar_Load_Processed.csv
```

数据策略：

- `date` 是 UTC timezone-aware 主时间轴，例如 `2023-01-01 00:00:00+0000`。
- 原始德国时间按 `Europe/Berlin` 解析，并显式处理 CET/CEST 夏令时。
- `month/day/weekday/hour` 从 UTC 转回 `Europe/Berlin` 本地时间后生成。
- processed 数据已通过检查：无重复 timestamp、严格 1 hour 连续、无 NaN/inf。

如需重新生成 processed 数据：

```bash
python data_surgery.py
```

源文件：

```text
Actual_generation_202301010000_202603011700_Hour (1).csv
Actual_consumption_202301010000_202603011700_Hour (2).csv
```

## Train / Val / Test 切分

当前使用 strict chronological split，不使用 random split，也不再使用简单比例切分。

| Split | UTC 起点 | UTC 终点 | 用途 |
|---|---|---|---|
| Train | `2023-01-01 00:00:00+00:00` | `2024-12-31 23:00:00+00:00` | 模型训练 |
| Validation | `2025-01-01 00:00:00+00:00` | `2025-03-31 23:00:00+00:00` | early stopping、epoch 选择、调参 |
| Test | `2025-04-01 00:00:00+00:00` | `2026-03-01 15:00:00+00:00` | 最终评估 |

严格窗口规则：

- `seq_x`、decoder label history、prediction target 必须全部位于同一个 split 内。
- Val 不回看 Train。
- Test 不回看 Val。
- 168h prediction target 不允许跨 split 边界。

详细说明见：

```text
docs/DATA_SPLIT_POLICY.md
```

## 环境

安装依赖：

```bash
pip install -r requirements.txt
```

本地已验证的 CPU 环境：

```text
D:\ProgramData\anaconda3\envs\torch_env\python.exe
torch 2.8.0+cpu
```

## 推荐训练参数

当前 168 -> 168 配置：

```text
--seq_len 168
--label_len 168
--pred_len 168
--features M
--enc_in 11
--dec_in 11
--c_out 11
```

输入输出形状：

- `batch_x` 形状为 `[batch, 168, 11]`。
- `batch_y` 形状为 `[batch, 336, 11]`，其中最后 168 步是监督 target。
- 前 3 个通道是 `Wind / Solar / Load`。
- 后 8 个通道是周期时间特征。

## CPU 小规模验证

正式上服务器前，可以先跑 CPU 小验证：

```bash
conda run -n torch_env python scripts/cpu_fedformer_168_smoke.py
```

这个脚本不是正式训练，只做 pipeline 验证：

- 使用 strict split。
- 只使用 train/val。
- 不在训练过程中调用 test。
- 最多 3 epoch，且每个 epoch 限制少量 batch。
- 保存 train/val loss。
- 保存 best checkpoint。
- 导出 validation forecast / actual / residual。
- 验证 `actual ~= forecast + residual`。

已验证结果见：

```text
docs/CPU_FEDFORMER_168_SMOKE_REPORT.md
```

输出目录：

```text
results/cpu_fedformer_168_smoke/
checkpoints/cpu_fedformer_168_smoke/checkpoint.pth
```

该目录不会覆盖旧的 `results/4.27`。

## 服务器训练命令

本项目当前可以选择两种 FEDformer 频域模块：

- `--version Fourier`：傅里叶增强模块，速度通常更快，依赖更少，建议作为第一次服务器正式跑的基线。
- `--version Wavelets`：小波增强模块，可能更适合局部突变和多尺度波动，但通常更慢，也更依赖环境兼容性。

建议服务器第一轮最规范地跑 **Fourier baseline**，确认完整训练、验证、测试和 `--full_inference` 导出都正常后，再用同样参数跑 Wavelets 做对照。

### 推荐正式训练：Fourier baseline

服务器较好时，建议从下面这套开始。它比 CPU smoke test 放大，但仍然比较稳：

- `batch_size=64`：如果显存不足，降到 `32`。
- `d_model=128`
- `d_ff=256`
- `modes=128`
- `e_layers=2`
- `d_layers=1`

```bash
python run.py --is_training 1 --task_id wind_solar_load_168_fourier --model FEDformer --version Fourier --mode_select random --modes 128 --data custom --root_path ./ --data_path Wind_Solar_Load_Processed.csv --features M --seq_len 168 --label_len 168 --pred_len 168 --enc_in 11 --dec_in 11 --c_out 11 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 256 --factor 1 --embed timeF --des strict_168_fourier --freq h --train_epochs 100 --patience 5 --early_stop_delta 0.00001 --batch_size 64 --learning_rate 0.0001 --lradj type3 --full_inference
```

如果服务器报 `ModuleNotFoundError: No module named 'exp'`，先确认当前目录是项目根目录，并且 `exp/exp_main.py` 已经上传：

```bash
pwd && ls -la exp exp/exp_main.py
```

如果文件存在但仍然无法导入，可以临时加上项目根目录到 `PYTHONPATH`：

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

然后再运行上面的单行训练命令。

### 可选对照训练：Wavelets

Wavelets 版除了 `--version`、`task_id`、`des` 外，其他参数建议先与 Fourier 保持一致，方便公平对比。

```bash
python run.py --is_training 1 --task_id wind_solar_load_168_wavelets --model FEDformer --version Wavelets --mode_select random --modes 128 --data custom --root_path ./ --data_path Wind_Solar_Load_Processed.csv --features M --seq_len 168 --label_len 168 --pred_len 168 --enc_in 11 --dec_in 11 --c_out 11 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 256 --factor 1 --embed timeF --des strict_168_wavelets --freq h --train_epochs 100 --patience 5 --early_stop_delta 0.00001 --batch_size 64 --learning_rate 0.0001 --lradj type3 --full_inference
```

### 参数放大是什么意思

- `batch_size`：一次送进模型的样本窗口数量。越大越吃显存，训练通常更稳定、更快；不改变模型能力。服务器正式建议先用 `64`，不够就降到 `32`、`16` 或 `8`。
- `d_model`：Transformer 内部隐藏维度，也就是每个时间步的表示宽度。越大模型容量越强，但显存和计算量明显增加。CPU smoke 用 `64`，服务器正式建议 `128` 起步。
- `d_ff`：前馈网络隐藏层宽度。通常和 `d_model` 同步放大，常见比例是 `2x` 到 `4x`。这里 `d_model=128` 配 `d_ff=256` 是比较稳的轻量正式配置。
- `modes`：FEDformer 在频域保留/采样的频率模式数量。越大能保留更多频率信息，可能更好捕捉波动，但计算更重，也可能引入噪声。正式建议 `128`；如果过慢可降到 `64`。
- `e_layers`：encoder 层数。越多模型更深，表达能力更强，也更慢、更容易过拟合。CPU smoke 用 `1`，正式建议 `2`。
- `d_layers`：decoder 层数。当前保持 `1`，先不要同时加深 decoder，避免变量太多。
- `--lradj type3`：固定学习率，不随 epoch 自动衰减。正式训练建议使用这个，避免 `type1` 每轮减半导致后期学习率过小。
- `--early_stop_delta 0.00001`：验证集 loss 至少下降 `1e-5` 才算真正改善；否则累计 patience。这样可以避免极小浮动反复重置早停。
- `--train_epochs 100 --patience 5`：最多跑 100 轮，但验证集连续 5 轮没有达到 `early_stop_delta` 要求的改善就停止。

推荐服务器第一轮：

```text
Fourier + d_model=128 + d_ff=256 + modes=128 + e_layers=2 + batch_size=64 + fixed lr + full_inference
```

如果显存不足，先只把 `batch_size` 从 `64` 降到 `32`。如果想继续提升模型容量，第二轮再考虑 `d_model=256`、`d_ff=512`，不要和版本切换同时改。

## 仅测试已有 checkpoint

测试时必须保持 `task_id`、模型结构参数、`des` 等与训练时一致，否则找不到对应 checkpoint。

Fourier 示例：

```bash
python run.py --is_training 0 --task_id wind_solar_load_168_fourier --model FEDformer --version Fourier --mode_select random --modes 128 --data custom --root_path ./ --data_path Wind_Solar_Load_Processed.csv --features M --seq_len 168 --label_len 168 --pred_len 168 --enc_in 11 --dec_in 11 --c_out 11 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 256 --factor 1 --embed timeF --des strict_168_fourier --freq h
```

Wavelets 测试时把 `task_id / --version / --des` 换成训练 Wavelets 时的值。

## 导出 train / val / test 预测和 residual

完整导出需要添加：

```bash
--full_inference
```

训练时已经加 `--full_inference` 的话，训练结束后会自动导出。若只想加载已有 checkpoint 导出，可以在 `--is_training 0` 命令后追加 `--full_inference`。

## 输出文件说明

训练和测试的标准输出目录格式：

```text
checkpoints/{setting}/checkpoint.pth
results/{setting}/metrics.npy
results/{setting}/pred.npy
results/{setting}/true.npy
results/{setting}/train_pred.npy
results/{setting}/train_true.npy
results/{setting}/train_res.npy
results/{setting}/val_pred.npy
results/{setting}/val_true.npy
results/{setting}/val_res.npy
results/{setting}/test_pred.npy
results/{setting}/test_true.npy
results/{setting}/test_res.npy
results/{setting}/train_forecast.npy
results/{setting}/train_actual.npy
results/{setting}/train_residual.npy
results/{setting}/val_forecast.npy
results/{setting}/val_actual.npy
results/{setting}/val_residual.npy
results/{setting}/test_forecast.npy
results/{setting}/test_actual.npy
results/{setting}/test_residual.npy
```

当前 residual 统一定义为：

```text
residual = actual - forecast
actual = forecast + residual
```

说明：

- `pred.npy` / `true.npy` 是 test 阶段输出。
- `train_pred.npy`、`val_pred.npy`、`test_pred.npy` 需要 `--full_inference`。
- `train_true.npy`、`val_true.npy`、`test_true.npy` 也会随 `--full_inference` 保存。
- `*_forecast.npy / *_actual.npy / *_residual.npy` 是含义更清晰的别名，内容分别对应 `*_pred.npy / *_true.npy / *_res.npy`。
- `*_res.npy` 和 `*_residual.npy` 都按 `actual - forecast` 理解。
- 老目录 `results/4.27` 是历史实验结果，不对应当前 strict split 和 168 输入配置。

训练日志和报告会保存在：

```text
experiments/{setting}_{timestamp}/config.json
experiments/{setting}_{timestamp}/train_log.json
experiments/{setting}_{timestamp}/metrics_test.json
experiments/{setting}_{timestamp}/figures/training_curves.pdf
experiments/{setting}_{timestamp}/summary.txt
```

其中 `train_log.json` 记录每个 epoch 的 `train_loss`、`val_loss`、学习率和时间戳；训练过程中不会记录 per-epoch `test_loss`。

## 当前已验证的 CPU 小实验

目录：

```text
results/cpu_fedformer_168_smoke/
```

关键结果：

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.524678 | 0.393704 |
| 2 | 0.362671 | 0.266078 |
| 3 | 0.266657 | 0.193069 |

导出数组：

```text
val_forecast.npy
val_actual.npy
val_residual.npy
val_pred.npy
val_true.npy
val_res.npy
```

重构误差：

```text
max(abs(forecast + residual - actual)) = 0.001953125
```

## 项目结构

```text
data_surgery.py                     # 从原始下载文件生成 UTC processed 数据
run.py                              # 主训练/测试入口
data_provider/                      # Dataset/DataLoader
exp/                                # 训练、验证、测试、导出逻辑
models/                             # FEDformer/Autoformer/Informer/Transformer
layers/                             # 模型层
utils/                              # 工具函数
scripts/cpu_fedformer_168_smoke.py  # CPU 小规模 168->168 pipeline 验证
docs/                               # 数据审计、split policy、smoke report
results/                            # 预测输出
checkpoints/                        # 模型 checkpoint
```

## 重要报告

```text
docs/DATA_PREPROCESSING_AUDIT.md
docs/DATA_PREPROCESSING_AUDIT_FIXED.md
docs/DATA_SPLIT_POLICY.md
docs/PIPELINE_SMOKE_TEST_FIXED.md
docs/CPU_FEDFORMER_168_SMOKE_REPORT.md
```

