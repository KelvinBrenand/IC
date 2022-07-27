from bn import BayesianNetwork
from sklearn import datasets
import numpy as np

bn1 = BayesianNetwork(datasets.load_iris().data[:30,:].tolist())
bn1.fit()

bn2 = BayesianNetwork(datasets.load_iris().data[50:80,:].tolist())
bn2.fit()

bn3 = BayesianNetwork(datasets.load_iris().data[100:130,:].tolist())
bn3.fit()

x_test = datasets.load_iris().data[30:50,:].tolist()+datasets.load_iris().data[80:100,:].tolist()+datasets.load_iris().data[130:,:].tolist()
y_test = ([0]*20)+([1]*20)+([2]*20)
y_pred = BayesianNetwork.predict(x_test,bn1,bn2,bn3)
print(np.array(BayesianNetwork.confusionMatrix(y_test, y_pred)))
print("Accuracy:",BayesianNetwork.accuracy(y_test, y_pred))