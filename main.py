from functions import kde, mle, nrm
import numpy as np
import matplotlib.pyplot as plt

DATA_SIZE = 1000
X_AMOUNT = 50
H_AMOUNT = 100
data = np.random.randn(DATA_SIZE).tolist()
#data = np.random.exponential(1.0, X_AMOUNT).tolist()
#data = np.random.laplace(0.0, 1.0, X_AMOUNT).tolist()
#h = np.linspace(0.01, 2, H_AMOUNT)
x = np.random.rand(X_AMOUNT)
x = np.sort(x)

# mle_result = mle(h, x, data)
# best = mle_result.index(max(mle_result))
# print("The best \'h\' is "+str(h[best])+".")
# plt.figure(1)
# plt.plot(h, mle_result, color='red')
# plt.ylabel('Maximum Likelihood Estimation (MLE)')
# plt.xlabel('\'h\' parameter')
# plt.show()


#best_h = h[best]
best_h = nrm(x, data, 2.0)
print("The best \'h\' is: "+str(best_h)+".")
xx = np.linspace(min(data), max(data), X_AMOUNT)
kde_result = []
for i in range(xx.shape[0]):
    kde_result.append(kde(xx[i], data, best_h))
plt.figure(2)
plt.hist(data, bins=25, density=True)
plt.plot(xx, kde_result, color='red')
plt.show()