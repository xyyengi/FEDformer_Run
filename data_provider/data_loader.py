import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class RevIN:
    """
    可逆实例归一化 (Reversible Instance Normalization)
    对每个样本独立进行归一化，保留实例级别的统计信息
    在推理时可以精确反归一化，避免跨样本统计信息泄露
    
    参考: Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, mode='instance'):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mode = mode
        if self.affine:
            self._init_params()
    
    def _init_params(self):
        self.affine_weight = np.ones(self.num_features)
        self.affine_bias = np.zeros(self.num_features)
    
    def forward(self, x, mode='norm'):
        """
        x: shape (seq_len, num_features) 或 (batch, seq_len, num_features)
        mode: 'norm' 进行归一化, 'denorm' 进行反归一化
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"模式 {mode} 不支持")
        return x
    
    def _get_statistics(self, x):
        # 计算每个实例的均值和标准差
        if len(x.shape) == 2:
            # (seq_len, num_features)
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.stdev = np.sqrt(np.var(x, axis=0, keepdims=True) + self.eps)
        else:
            # (batch, seq_len, num_features)
            self.mean = np.mean(x, axis=1, keepdims=True)
            self.stdev = np.sqrt(np.var(x, axis=1, keepdims=True) + self.eps)
    
    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self.stdev + self.mean
        return x


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date, utc=True)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 将周期编码后的时间特征(8维)与数据特征(3维)拼接，得到11维
        # data_stamp 是经过 sin/cos 编码的 8 维时间特征
        # data 是 3 维数据特征 (Wind, Solar, Load)
        combined_data = np.concatenate([data, data_stamp], axis=1)  # 拼接为 11 维
        
        self.data_x = combined_data[border1:border2]
        self.data_y = combined_data[border1:border2]
        self.data_stamp = data_stamp  # 保持时间标记用于 embedding

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        反归一化，只对数据特征（前3维）进行反归一化
        时间特征（后8维）保持不变，因为它们没有被 scaler 处理
        
        重要：对 Solar (索引1) 应用物理约束，确保 >= 0
        """
        original_shape = data.shape
        data_2d = data.reshape(-1, original_shape[-1])
        
        # 只对前 n_data_features 维进行反归一化
        n_data_features = self.scaler.n_features_in_  # scaler 处理的特征数量（3维）
        
        # 分离数据特征和时间特征
        data_features = data_2d[:, :n_data_features]
        time_features = data_2d[:, n_data_features:]
        
        # 只对数据特征反归一化
        data_features_inv = self.scaler.inverse_transform(data_features)
        
        # 物理约束：Solar (索引1) 不能为负值
        # 原因：MinMaxScaler 的 data_min_[Solar] = 0
        # 如果模型在归一化空间输出负值，反归一化后仍为负值
        solar_idx = 1
        data_features_inv[:, solar_idx] = np.clip(data_features_inv[:, solar_idx], 0, None)
        
        # 重新拼接
        result = np.concatenate([data_features_inv, time_features], axis=1)
        return result.reshape(original_shape)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    SPLIT_RANGES_UTC = {
        'train': (
            pd.Timestamp('2023-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2024-12-31 23:00:00', tz='UTC'),
        ),
        'val': (
            pd.Timestamp('2025-01-01 00:00:00', tz='UTC'),
            pd.Timestamp('2025-03-31 23:00:00', tz='UTC'),
        ),
        'test': (
            pd.Timestamp('2025-04-01 00:00:00', tz='UTC'),
            pd.Timestamp('2026-03-01 15:00:00', tz='UTC'),
        ),
    }

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_cycle_time_enc=True, use_revin=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_cycle_time_enc = use_cycle_time_enc  # 周期性时间编码开关
        self.use_revin = use_revin  # RevIN归一化开关
        self.n_data_features = 3  # 数据特征数量 (Wind, Solar, Load)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _validate_processed_time_axis(self, df_raw):
        if 'date' not in df_raw.columns:
            raise ValueError("Dataset_Custom requires a date column")

        date_text = df_raw['date'].astype(str)
        has_explicit_tz = date_text.str.contains(r'(Z|[+-]\d{2}:?\d{2})$', regex=True).all()
        if not has_explicit_tz:
            raise ValueError(
                "Processed data must use timezone-aware UTC timestamps in date, "
                "for example 2023-01-01 00:00:00+0000"
            )

        dates_utc = pd.to_datetime(df_raw['date'], utc=True, errors='raise')
        if str(dates_utc.dt.tz) != 'UTC':
            raise ValueError("date column must parse as UTC")
        if not dates_utc.is_monotonic_increasing:
            raise ValueError("date column must be monotonic increasing")
        if dates_utc.duplicated().any():
            duplicated = dates_utc[dates_utc.duplicated()].head().tolist()
            raise ValueError(f"duplicate UTC timestamps found: {duplicated}")

        deltas = dates_utc.diff().dropna()
        bad_deltas = deltas[deltas != pd.Timedelta(hours=1)]
        if not bad_deltas.empty:
            examples = [
                (dates_utc.iloc[i - 1], dates_utc.iloc[i], bad_deltas.loc[i])
                for i in bad_deltas.index[:5]
            ]
            raise ValueError(f"non-1h UTC timestamp deltas found: {examples}")

        if df_raw.isna().any().any():
            raise ValueError("NaN values found in processed data")
        numeric = df_raw.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("inf values found in processed data")

        return dates_utc

    def _get_strict_split_bounds(self, dates_utc, split_name):
        split_start, split_end = self.SPLIT_RANGES_UTC[split_name]
        mask = (dates_utc >= split_start) & (dates_utc <= split_end)
        if not mask.any():
            raise ValueError(f"{split_name}: no rows found for configured UTC split range")

        indices = np.flatnonzero(mask.to_numpy())
        border1 = int(indices[0])
        border2 = int(indices[-1]) + 1
        if dates_utc.iloc[border1] != split_start:
            raise ValueError(f"{split_name}: missing split start timestamp {split_start}")
        if dates_utc.iloc[border2 - 1] != split_end:
            raise ValueError(f"{split_name}: missing split end timestamp {split_end}")

        split_len = border2 - border1
        min_len = self.seq_len + self.pred_len
        if split_len < min_len:
            raise ValueError(f"{split_name}: split length {split_len} < seq_len + pred_len {min_len}")

        return border1, border2, split_start, split_end

    def _build_window_info(self, dates_utc, border1, index):
        s_begin = border1 + index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        target_begin = s_end
        target_end = r_end

        target_dates = dates_utc.iloc[target_begin:target_end]
        if len(target_dates) != self.pred_len:
            raise ValueError(f"{self.flag}: target length {len(target_dates)} != pred_len {self.pred_len}")
        target_deltas = target_dates.diff().dropna()
        if not (target_deltas == pd.Timedelta(hours=1)).all():
            raise ValueError(f"{self.flag}: target timestamps are not strictly hourly")

        return {
            'seq_x': (dates_utc.iloc[s_begin], dates_utc.iloc[s_end - 1]),
            'label': (dates_utc.iloc[r_begin], dates_utc.iloc[s_end - 1]),
            'target': (dates_utc.iloc[target_begin], dates_utc.iloc[target_end - 1]),
        }

    def _validate_strict_windows(self, dates_utc, border1, border2, split_start, split_end):
        max_index = (border2 - border1) - self.seq_len - self.pred_len
        first = self._build_window_info(dates_utc, border1, 0)
        last = self._build_window_info(dates_utc, border1, max_index)

        for sample_name, sample_info in [('first', first), ('last', last)]:
            for window_name, (start, end) in sample_info.items():
                if start < split_start or end > split_end:
                    raise ValueError(
                        f"{self.flag}: {sample_name} {window_name} crosses split boundary: "
                        f"{start} to {end}, split {split_start} to {split_end}"
                    )

        self.split_info = {
            'name': self.flag,
            'split_start': split_start,
            'split_end': split_end,
            'raw_points': border2 - border1,
            'valid_windows': max_index + 1,
            'first_sample': first,
            'last_sample': last,
            'cross_split_windows': False,
            'timestamp_discontinuous': False,
            'has_nan_or_inf': False,
        }

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        dates_utc = self._validate_processed_time_axis(df_raw)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # 对于 M 模式（多变量预测多变量），target 参数不重要
        # 只需要确保 date 列被正确处理
        cols = list(df_raw.columns)
        if 'date' in cols:
            cols.remove('date')
        # 只有在 target 存在且不是多变量模式时才移除
        if self.features != 'M' and self.target in cols:
            cols.remove(self.target)
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            df_raw = df_raw[['date'] + cols]
        border1, border2, split_start, split_end = self._get_strict_split_bounds(dates_utc, self.flag)
        self._validate_strict_windows(dates_utc, border1, border2, split_start, split_end)

        time_cols = ['month', 'day', 'weekday', 'hour']
        if self.features == 'M' or self.features == 'MS':
            cols_data = [c for c in df_raw.columns[1:] if c not in time_cols]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_start, train_end = self.SPLIT_RANGES_UTC['train']
            train_mask = (dates_utc >= train_start) & (dates_utc <= train_end)
            train_data = df_data[train_mask.to_numpy()]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 先对完整数据生成时间特征，然后再切片（确保行数一致）
        df_stamp = df_raw[['date'] + [c for c in time_cols if c in df_raw.columns]]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if all(c in df_stamp.columns for c in time_cols):
            # Use provided external time features directly
            raw_time_data = df_stamp[time_cols].values
            if self.use_cycle_time_enc:
                # 应用周期性编码 (Sin/Cos变换)
                data_stamp = self._apply_cyclical_encoding(raw_time_data, time_cols)
            else:
                data_stamp = raw_time_data
        else:
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                raw_time_data = df_stamp.drop(['date'], 1).values
                if self.use_cycle_time_enc:
                    data_stamp = self._apply_cyclical_encoding(raw_time_data, time_cols)
                else:
                    data_stamp = raw_time_data
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

        # 将周期编码后的时间特征(8维)与数据特征(3维)拼接，得到11维
        # data_stamp 是经过 sin/cos 编码的 8 维时间特征（完整数据）
        # data 是 3 维数据特征 (Wind, Solar, Load)（完整数据）
        combined_data = data
        
        # 切片取对应的数据集部分
        self.data_x = combined_data[border1:border2]
        self.data_y = combined_data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]  # 时间标记也切片
        self.dates_utc = dates_utc.iloc[border1:border2].reset_index(drop=True)
        self.border1 = border1
        self.border2 = border2

        print(
            f"[SPLIT] {self.flag}: {split_start} -> {split_end}, "
            f"points={self.split_info['raw_points']}, windows={self.split_info['valid_windows']}"
        )
        print(
            f"[SPLIT] {self.flag} first target: "
            f"{self.split_info['first_sample']['target'][0]} -> {self.split_info['first_sample']['target'][1]}"
        )
        print(
            f"[SPLIT] {self.flag} last target: "
            f"{self.split_info['last_sample']['target'][0]} -> {self.split_info['last_sample']['target'][1]}"
        )

    def _apply_cyclical_encoding(self, data, col_names):
        """
        对时间特征应用周期性编码 (Sin/Cos变换)
        使模型能够理解时间的周期性：hour=23 和 hour=0 是相邻的
        """
        encoded_features = []
        for i, col in enumerate(col_names):
            feature = data[:, i]
            if col == 'hour':
                # 24小时周期
                sin_val = np.sin(2 * np.pi * feature / 24)
                cos_val = np.cos(2 * np.pi * feature / 24)
                encoded_features.extend([sin_val, cos_val])
            elif col == 'weekday':
                # 7天周期
                sin_val = np.sin(2 * np.pi * feature / 7)
                cos_val = np.cos(2 * np.pi * feature / 7)
                encoded_features.extend([sin_val, cos_val])
            elif col == 'month':
                # 12个月周期
                sin_val = np.sin(2 * np.pi * feature / 12)
                cos_val = np.cos(2 * np.pi * feature / 12)
                encoded_features.extend([sin_val, cos_val])
            elif col == 'day':
                # 31天周期 (近似)
                sin_val = np.sin(2 * np.pi * feature / 31)
                cos_val = np.cos(2 * np.pi * feature / 31)
                encoded_features.extend([sin_val, cos_val])
            else:
                # 其他特征保持原样
                encoded_features.append(feature)
        
        return np.stack(encoded_features, axis=1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        反归一化，只对数据特征（前3维）进行反归一化
        时间特征（后8维）保持不变，因为它们没有被 scaler 处理
        
        重要：对 Solar (索引1) 应用物理约束，确保 >= 0
        """
        original_shape = data.shape
        data_2d = data.reshape(-1, original_shape[-1])
        
        # 只对前 n_data_features 维进行反归一化
        n_data_features = self.scaler.n_features_in_  # scaler 处理的特征数量（3维）
        
        # 分离数据特征和时间特征
        data_features = data_2d[:, :n_data_features]
        time_features = data_2d[:, n_data_features:]
        
        # 只对数据特征反归一化
        data_features_inv = self.scaler.inverse_transform(data_features)
        
        # 物理约束：Solar (索引1) 不能为负值
        # 原因：MinMaxScaler 的 data_min_[Solar] = 0
        # 如果模型在归一化空间输出负值，反归一化后仍为负值
        solar_idx = 1
        data_features_inv[:, solar_idx] = np.clip(data_features_inv[:, solar_idx], 0, None)
        
        # 重新拼接
        return data_features_inv.reshape(original_shape)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
