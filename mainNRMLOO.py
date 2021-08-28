from functions import functions
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=123)
DATA_SIZE = 300

data = np.random.randn(DATA_SIZE).tolist()+(np.random.randn(DATA_SIZE)+4).tolist()
obj = functions()

best_h = obj.newtonRaphsonLeaveOneOut(data, 1.0)
kde_result = []
databkp = data.copy()
for i in range(len(data)):
    element = databkp[i]
    databkp.pop(i)
    kde_result.append(obj.kernelDensityEstimation(element, databkp, best_h))
    databkp.clear()
    databkp = data.copy()

sorted_kde_result = [x for _,x in sorted(zip(data, kde_result))]

plt.figure(1)
plt.hist(data, bins=25, density=True)
plt.plot(sorted(data), sorted_kde_result, color='red')
plt.title("Newton-Raphson Method using \'Leave-one-out\'\nBest \'h\' = %.5f" % best_h)
plt.show()