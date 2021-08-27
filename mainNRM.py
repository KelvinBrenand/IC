from functions import functions
import numpy as np
import matplotlib.pyplot as plt
import time

DATA_SIZE = 1000
X_AMOUNT = 50
np.random.seed(seed=123)

data = np.random.randn(DATA_SIZE).tolist()+(np.random.randn(DATA_SIZE)*np.sqrt(1)+3).tolist()
#data = np.random.exponential(1.0, DATA_SIZE).tolist()
#data = np.random.laplace(0.0, 1.0, DATA_SIZE).tolist()
maxx = max(data)
minn = min(data)
x = np.random.rand(X_AMOUNT)*(maxx-minn)+minn
obj = functions()

tempoi = time.time()
best_h = obj.nrm(x, data, 1.0)
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
plt.title("Newton-Raphson Method\nInitial \'h\' = 1. Best \'h\' = "+str(best_h))
plt.show()