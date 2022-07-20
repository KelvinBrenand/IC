from sklearn import datasets
from bn import bayesianNetwork
import numpy as np

data = datasets.load_iris().data[:,:4]
obj = bayesianNetwork()
print("Input shape: ",data.shape)
print(np.array(obj.MLE(data.tolist())))