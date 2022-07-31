from bn import BayesianNetwork
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv('BayesianNetwork\Iris.csv')
X = df.iloc[:, [1, 2, 3, 4]].values.tolist()
y = df.iloc[:, 5].values.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0, stratify=y)
classes = list(set(y))

networks = []
for j in classes:
    networks.append(BayesianNetwork([X_train[i] for i in range(len(y_train)) if y_train[i] == j]))

y_pred = BayesianNetwork.predict(X_test, classes, networks)
print(np.array(BayesianNetwork.confusionMatrix(y_test, y_pred)))
print("Accuracy:",BayesianNetwork.accuracy(y_test, y_pred))