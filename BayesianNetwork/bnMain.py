from sklearn import datasets
from bn import newtonRapson
import numpy as np

data = datasets.load_wine().data[:,:4]
obj = newtonRapson()
print("Input shape: ",data.shape)
print(np.array(obj.MLE(data.tolist())))