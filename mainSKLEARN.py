from functions import functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth
import time

DATA_SIZE = 1000
X_AMOUNT = 500
np.random.seed(seed=123)
data = np.random.randn(DATA_SIZE).tolist()
#data = np.random.exponential(1.0, DATA_SIZE).tolist()
#data = np.random.laplace(0.0, 1.0, DATA_SIZE).tolist()
obj = functions()
arr = np.array(data)
arr = np.reshape(arr, (1000, 1))

tempoi = time.time()
best_h = estimate_bandwidth(arr, quantile=0.1)
tempototal = time.time() - tempoi
print("The best \'h\' is: "+str(best_h)+".")
print("Time: "+str(tempototal)+"s.")
xx = np.linspace(min(data), max(data), X_AMOUNT)
kde_result = []
for i in range(xx.shape[0]):
    kde_result.append(obj.kde(xx[i], data, best_h))
plt.figure(1)
plt.hist(data, bins=25, density=True)
plt.plot(xx, kde_result, color='red')
plt.title("Sklearn Method\nBest \'h\' = "+str(best_h))
plt.show()