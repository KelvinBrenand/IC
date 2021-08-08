from functions import kde, mle, nrm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import estimate_bandwidth
import time

DATA_SIZE = 1000
X_AMOUNT = 50
H_AMOUNT = 100

data = np.random.randn(DATA_SIZE).tolist()
#data = np.random.exponential(1.0, DATA_SIZE).tolist()
#data = np.random.laplace(0.0, 1.0, DATA_SIZE).tolist()
x = np.random.rand(X_AMOUNT)

# h = np.linspace(0.01, 2, H_AMOUNT) #mle
# x = np.sort(x)                     #mle

tempoi = time.time()
#best_h = mle(h, x, data)
best_h = nrm(x, data, 1.0)

# arr = np.array(data)
# arr = np.reshape(arr, (1000, 1))
# best_h = estimate_bandwidth(arr, quantile=0.05)
tempototal = time.time() - tempoi
print("The best \'h\' is: "+str(best_h)+".")
print("Time: "+str(tempototal)+"s.")
xx = np.linspace(min(data), max(data), X_AMOUNT)
kde_result = []
for i in range(xx.shape[0]):
    kde_result.append(kde(xx[i], data, best_h))
plt.figure(2)
plt.hist(data, bins=25, density=True)
plt.plot(xx, kde_result, color='red')
plt.title("Newton-Raphson Method\nInitial \'h\' = 1. Best \'h\' = "+str(best_h))
plt.show()