from functions import functions
import numpy as np

obj = functions()
DATA_SIZE = 1000
X_AMOUNT = 50
mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean, cov, DATA_SIZE)
x = np.random.multivariate_normal(mean, cov, X_AMOUNT)
best_h = 2.0
kde_result = []
for i in range(len(x)):
    kde_result.append(obj.multidimensionalKernelDensityEstimation(x[i], data, best_h))

kdeFinal = np.array(kde_result)
print(kdeFinal, kdeFinal.shape)