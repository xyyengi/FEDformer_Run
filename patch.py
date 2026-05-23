with open("data_provider/data_loader.py", "r", encoding="utf-8") as f:
    text = f.read()

s1 = """        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]"""

s2 = """        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if all(c in df_stamp.columns for c in time_cols):
            # Use provided external time features directly
            data_stamp = df_stamp[time_cols].values
        else:
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]"""

import sys
if s1 in text:
    idx = text.find(s1)
    text = text[:idx] + s2 + text[idx+len(s1):] # Only replace the first occurrence which is in Dataset_Custom
    with open("data_provider/data_loader.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Patched.")
else:
    print("Not found.")
