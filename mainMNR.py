from functions import functions
import numpy as np

obj = functions()
DATA_SIZE = 100
mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean, cov, DATA_SIZE)
#best_h = obj.multidimentionalNewtonRaphson(data, 1.0)
best_h = 1.0
kde_result = obj.multidimensionalKernelDensityEstimation(data, best_h)
kdeFinal = np.array(kde_result)
print(kdeFinal, kdeFinal.shape)