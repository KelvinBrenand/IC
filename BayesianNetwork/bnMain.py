import numpy as np
import pandas as pd
from datetime import datetime
from bn import BayesianNetwork

df = pd.read_csv('BayesianNetwork\wifi_localization.csv')
df = df.dropna()
X = df.iloc[:, :-1].values.tolist() #[:, 1:-1] Breast Cancer
y = df.iloc[:, -1].values.tolist()

start_time = datetime.now()
networks, acc, confMtx = BayesianNetwork.kfoldcv(X, y, k=10, num_h=10, accuracy=True, confMtx=True)
end_time = datetime.now()

print('Duration: {}'.format(end_time - start_time))
print("Accuracy:",acc)
print(np.array(confMtx),"\n")