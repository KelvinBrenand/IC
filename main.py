from KDE import kde
import numpy as np
import matplotlib.pyplot as plt
import time

DATA_SIZE = 1000
X_AMOUNT = 50
H_AMOUNT = 100
data = np.random.randn(DATA_SIZE).tolist()
h = np.linspace(0.01, 5, H_AMOUNT)
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
print(best_h)
x_result = (x_result*6)-3
y_result = []
for i in range(x_result.shape[0]):
    y_result.append(kde(x_result[i], data, best_h))
plt.figure(2)
plt.hist(data, bins=X_AMOUNT, density=True)
plt.plot(x_result, y_result, color='red')
plt.show() 