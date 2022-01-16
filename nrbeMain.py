from nrbe import newtonRapson, multivariateNewtonRapson
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=123)
DATA_SIZE = 100
#reValue = 10

#1 Dimensional
# data = np.random.randn(DATA_SIZE).tolist()+(np.random.randn(DATA_SIZE)+4).tolist()
# obj = newtonRapson()
# best_h = obj.newtonRaphson(data, 1.0)
# kde_result = obj.kernelDensityEstimation(data, best_h)
# sorted_kde_result = [x for _,x in sorted(zip(data, kde_result))]

# plt.figure(1)
# plt.hist(data, bins=25, density=True)
# plt.plot(sorted(data), sorted_kde_result, color='red')
# plt.title("Newton-Raphson Method\nBandwidth parameter \'h\' = %.5f" % best_h)
# plt.show()

# N Dimensional
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# data = np.random.multivariate_normal(mean, cov, DATA_SIZE)
# x, y = data.T
# data = data.tolist()
# obj = multivariateNewtonRapson()
# best_h = obj.multivariateNewtonRaphson(data)
# kde_result = obj.multivariateKernelDensityEstimation(data, best_h)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatt = ax.scatter(x, y, kde_result, c=np.abs(kde_result), cmap=plt.get_cmap("viridis"), marker='o')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('KDE')
# plt.title('Multivariate Newton-Raphson Method')
# fig.colorbar(scatt, shrink=0.5,location="left", aspect=7)
# plt.show()

# x = np.reshape(x, (reValue, reValue))
# y = np.reshape(y, (reValue, reValue))
# z = np.reshape(kde_result, (reValue, reValue))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap="viridis", antialiased=False)
# ax.set_zlim(0, 0.2)
# ax.zaxis.set_major_locator(plt.LinearLocator(10))
# ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('KDE')
# plt.title('Multidimensional Newton-Raphson Method')
# fig.colorbar(surf, shrink=0.5,location="left", aspect=7)
# plt.show()