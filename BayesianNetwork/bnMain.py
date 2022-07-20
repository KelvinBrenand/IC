from sklearn import datasets
from bn import bayesianNetwork
import numpy as np

obj = bayesianNetwork()
#data = datasets.load_iris().data[:,:4]
#print("Input shape: ",data.shape)
#print(np.array(obj.MLE(data.tolist())))

data1 = datasets.load_iris().data[:50,:]
print(np.array(obj.MLE(data1.tolist())))

data1 = datasets.load_iris().data[50:100,:]
print(np.array(obj.MLE(data1.tolist())))

data1 = datasets.load_iris().data[100:,:]
print(np.array(obj.MLE(data1.tolist())))