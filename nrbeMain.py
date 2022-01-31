from nrbe import newtonRapson, multivariateNewtonRapson
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=123)
DATA_SIZE = 100

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
mean = [0, 0]
cov = [[1, 0], 
        [0, 1]]

mean2 = [2, 2]
cov2 = [[1, -0.8], 
        [-0.8, 1]]

data1 = np.random.multivariate_normal(mean, cov, DATA_SIZE)
data2 = np.random.multivariate_normal(mean2, cov2, DATA_SIZE)
data = np.concatenate((data1,data2))

x, y = data.T
data = data.tolist()
obj = multivariateNewtonRapson()
best_h = obj.multivariateNewtonRaphson(data)
kde_result = obj.multivariateKernelDensityEstimation(data, best_h)
print("\nh = "+str(best_h)+"\n")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatt = ax.scatter(x, y, kde_result, c=np.abs(kde_result), cmap=plt.get_cmap("viridis"), marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('KDE')
plt.title('Multivariate Newton-Raphson Method')
fig.colorbar(scatt, shrink=0.5,location="left", aspect=7)
plt.show()
