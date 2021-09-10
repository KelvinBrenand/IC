from functions import functions
from matplotlib import pyplot as plt
from operator import itemgetter
import numpy as np
DATA_SIZE = 49
reValue = 7
np.random.seed(seed=123)
obj = functions()

mean = [0, 0]
cov = [[1, 0], [0, 1]]
data = np.random.multivariate_normal(mean, cov, DATA_SIZE)
x, y = data.T
best_h = obj.multidimensionalNewtonRaphson(data, 1.0)
kde_result = obj.multidimensionalKernelDensityEstimation(data, best_h)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatt = ax.scatter(x, y, kde_result, c=np.abs(kde_result), cmap=plt.get_cmap("viridis"), marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('KDE')
plt.title('Multidimensional Newton-Raphson Method')
fig.colorbar(scatt, shrink=0.5,location="left", aspect=7)
plt.show()

x = np.reshape(x, (reValue, reValue))
y = np.reshape(y, (reValue, reValue))
z = np.reshape(kde_result, (reValue, reValue))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap="viridis", antialiased=False)
ax.set_zlim(0, 0.2)
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('KDE')
plt.title('Multidimensional Newton-Raphson Method')
fig.colorbar(surf, shrink=0.5,location="left", aspect=7)
plt.show()