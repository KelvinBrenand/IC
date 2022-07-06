from sklearn import datasets
from bn import newtonRapson
import numpy as np

data = datasets.load_iris().data[:,:3]
obj = newtonRapson()
print(data.shape)
print(np.array(obj.MLE(data.tolist())))