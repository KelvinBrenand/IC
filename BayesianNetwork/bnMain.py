from sklearn import datasets
from bn import BayesianNetwork
import numpy as np

data1 = datasets.load_iris().data[:50,:].tolist()
bn1 = BayesianNetwork(data1)
bn1.fit()
print(np.array(bn1.adjacencyMatrix))
#bn1.save('model.pkl')

#bn1 = BayesianNetwork.load('model.pkl')
#print(np.array(bn1.adjacencyMatrix))


#data2 = datasets.load_iris().data[50:100,:].tolist()
#bn2 = BayesianNetwork(data2)

#data3 = datasets.load_iris().data[100:,:].tolist()
#bn3 = BayesianNetwork(data3)