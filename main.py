from KDE import kde
import numpy as np
import matplotlib.pyplot as plt
import time
DATA_SIZE = 10000
X_AMOUNT = 50

data = np.random.randn(DATA_SIZE).tolist()
h = 1.06*np.std(data)*(DATA_SIZE**-0.2)
x_result = np.linspace(min(data), max(data), X_AMOUNT)
y_result = []
tempoi = time.time()
for i in range(x_result.shape[0]):
    y_result.append(kde(x_result[i], data, h))
tempototal = time.time() - tempoi
print(tempototal)
plt.figure(1)
plt.hist(data, bins=X_AMOUNT, density=True)
plt.plot(x_result, y_result, color='red')
plt.show()