from sklearn import datasets
from bn import BayesianNetwork
import numpy as np

data1 = datasets.load_iris().data[:50,:]
bn1 = BayesianNetwork(data1)
bn1.fit()
print(bn1.graphProbabilities())
print(np.array(bn1.adjacencyMatrix()))
#bn1.save('model.pkl')
#bn1 = BayesianNetwork.load('model.pkl')

#data2 = datasets.load_iris().data[50:100,:]
#bn2 = BayesianNetwork(data2)

#data3 = datasets.load_iris().data[100:,:]
#bn3 = BayesianNetwork(data3)