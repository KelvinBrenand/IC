from bn import BayesianNetwork
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('BayesianNetwork\Iris.csv')
X = df.iloc[:, [1, 2, 3, 4]].values.tolist()
y = df.iloc[:, 5].values.tolist()
classes = list(set(y))

networks = BayesianNetwork.kfoldcv(X, classes, 2)