from sklearn import datasets
from bn import BayesianNetwork
import numpy as np

data1 = datasets.load_iris().data[:50,:].tolist()
bn1 = BayesianNetwork(data1)
bn1.fit()
#print(np.array(bn1.adjacencyMatrix))
#bn1.save('model.pkl')
#bn1 = BayesianNetwork.load('model.pkl')


data2 = datasets.load_iris().data[50:100,:].tolist()
bn2 = BayesianNetwork(data2)
bn2.fit()

data3 = datasets.load_iris().data[100:,:].tolist()
bn3 = BayesianNetwork(data3)
bn3.fit()

#dado = [5.1, 3.5, 1.4, 0.2] #Classe1
#dado = [7.0, 3.2, 4.7, 1.4] #Classe2
dado = [6.3, 3.3, 6.0, 2.5] #Classe3
print(bn1.predict(dado))
print(bn2.predict(dado))
print(bn3.predict(dado))