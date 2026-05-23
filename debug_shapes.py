import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.DataFrame(np.random.randn(10,3), columns=['A','B','C'])
train_data = df.values
scaler.fit(train_data)

preds = np.random.randn(10, 3)
i_preds = scaler.inverse_transform(preds)
print("Worked!", i_preds.shape)
