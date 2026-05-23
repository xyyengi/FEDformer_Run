import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('D:/FEDformer_Run/Wind_Solar_Load_Processed.csv')
print('CSV columns:', df.columns.tolist())
print('CSV shape:', df.shape)

# 检查时间列
time_cols = ['month', 'day', 'weekday', 'hour']
cols_data = [c for c in df.columns[1:] if c not in time_cols]
print('Data columns (物理量):', cols_data)

df_data = df[cols_data]
print('df_data shape:', df_data.shape)

# 模拟 scaler
scaler = MinMaxScaler(feature_range=(0, 1))
num_train = int(len(df) * 0.7)
train_data = df_data[:num_train]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)
print('Scaled data shape:', data.shape)
print('Scaled data range:', data.min(), data.max())

# 模拟时间特征编码
raw_time_data = df[time_cols].values
print('Raw time data shape:', raw_time_data.shape)

# Sin/Cos 编码
encoded_features = []
for i, col in enumerate(time_cols):
    feature = raw_time_data[:, i]
    if col == 'hour':
        sin_val = np.sin(2 * np.pi * feature / 24)
        cos_val = np.cos(2 * np.pi * feature / 24)
        encoded_features.extend([sin_val, cos_val])
    elif col == 'weekday':
        sin_val = np.sin(2 * np.pi * feature / 7)
        cos_val = np.cos(2 * np.pi * feature / 7)
        encoded_features.extend([sin_val, cos_val])
    elif col == 'month':
        sin_val = np.sin(2 * np.pi * feature / 12)
        cos_val = np.cos(2 * np.pi * feature / 12)
        encoded_features.extend([sin_val, cos_val])
    elif col == 'day':
        sin_val = np.sin(2 * np.pi * feature / 31)
        cos_val = np.cos(2 * np.pi * feature / 31)
        encoded_features.extend([sin_val, cos_val])

data_stamp = np.stack(encoded_features, axis=1)
print('Time stamp shape (Sin/Cos编码后):', data_stamp.shape)
print('Time stamp range:', data_stamp.min(), data_stamp.max())

# 最终拼接
combined = np.concatenate([data, data_stamp], axis=1)
print('Combined shape:', combined.shape)
print('Combined前3维(物理量) range:', combined[:, :3].min(), combined[:, :3].max())
print('Combined后8维(时间) range:', combined[:, 3:].min(), combined[:, 3:].max())

# 验证 scaler 只作用于物理量
print('\n=== 验证 Scaler 只作用于物理量 ===')
print('scaler.n_features_in_:', scaler.n_features_in_)
print('scaler.n_samples_seen_:', scaler.n_samples_seen_)
print('scaler.data_min_:', scaler.data_min_)
print('scaler.data_max_:', scaler.data_max_)