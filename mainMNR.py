from functions import functions
from matplotlib import pyplot as plt
import numpy as np
DATA_SIZE = 250
np.random.seed(seed=123)
obj = functions()

mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean, cov, DATA_SIZE)
best_h = obj.multidimentionalNewtonRaphson(data, 1.0)
kde_result = obj.multidimensionalKernelDensityEstimation(data, best_h)
print(np.array(kde_result).shape)

############################################################################
# X,Y = np.meshgrid(np.linspace(-5, 5, 25), np.linspace(-5, 5, 25))
# Z = np.array(kde_result)

# fig = plt.figure(1)
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", antialiased=False)
# plt.show()