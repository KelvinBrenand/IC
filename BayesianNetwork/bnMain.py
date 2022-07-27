from bn import BayesianNetwork
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('BayesianNetwork\Iris.csv')
X = df.iloc[:, [1, 2, 3, 4]].values.tolist()
y = df.iloc[:, 5].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0, stratify=y)
classes = list(set(y))

bn1 = BayesianNetwork([X_train[i] for i in range(len(y_train)) if y_train[i] == classes[0]])
bn1.fit()
bn2 = BayesianNetwork([X_train[i] for i in range(len(y_train)) if y_train[i] == classes[1]])
bn2.fit()
bn3 = BayesianNetwork([X_train[i] for i in range(len(y_train)) if y_train[i] == classes[2]])
bn3.fit()

y_pred = BayesianNetwork.predict(X_test, classes, bn1, bn2, bn3)
print(np.array(BayesianNetwork.confusionMatrix(y_test, y_pred)))
print("Accuracy:",BayesianNetwork.accuracy(y_test, y_pred))