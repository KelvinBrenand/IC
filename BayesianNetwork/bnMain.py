from bn import BayesianNetwork
import numpy as np
import pandas as pd

df = pd.read_csv('BayesianNetwork\Iris.csv')
X = df.iloc[:, [1, 2, 3, 4]].values.tolist()
y = df.iloc[:, 5].values.tolist()

networks, acc, confMtx = BayesianNetwork.kfoldcv(X, y, 2, accuracy=True, confMtx=True)
print("Accuracy:",acc)
print(np.array(confMtx))