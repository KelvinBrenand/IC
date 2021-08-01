from KDE import kde
import numpy as np
import matplotlib.pyplot as plt
import time

DATA_SIZE = 1000
X_AMOUNT = 500
H_AMOUNT = 100
data = np.random.randn(DATA_SIZE).tolist()
#data = np.random.exponential(1.0, X_AMOUNT).tolist()
#data = np.random.laplace(0.0, 1.0, X_AMOUNT).tolist()
h = np.linspace(0.01, 2, H_AMOUNT)
x_result = np.random.rand(X_AMOUNT)
x_result = np.sort(x_result)
auxVar = [None] * X_AMOUNT
mle = []
tempoi = time.time()
for i in range(h.shape[0]):
    for j in range(x_result.shape[0]):
        auxVar[j] = np.log(kde(x_result[j], data, h[i]))
    mle.append(np.sum(auxVar))
tempototal = time.time() - tempoi
print("Time: "+str(tempototal)+"s")
maxMLE = max(mle)
best = mle.index(maxMLE)
print("The best \'h\' is "+str(h[best])+", which gives the MLE value of "+str(maxMLE)+".")
plt.figure(1)
plt.plot(h, mle, color='red')
plt.ylabel('Maximum Likelihood Estimation (MLE)')
plt.xlabel('\'h\' parameter')
plt.show()

best_h = h[best]
print("h: "+str(best_h))
#x = (x*6)-3  #normal
#x = (x*5)-0  #exponencial
#x = (x*16)-8 #laplace
#x = (x*max(data)*2)-max(data)
xx = np.linspace(min(data), max(data), X_AMOUNT)
kde_result = []
for i in range(xx.shape[0]):
    kde_result.append(kde(xx[i], data, best_h))
plt.figure(1)
plt.hist(data, bins=25, density=True)
plt.plot(xx, kde_result, color='red')
plt.show() 