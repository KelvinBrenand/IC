import math
import numpy as np

def gaussian(x):
    return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
def kde(x, array, h):
    N = len(array)
    sum = 0
    for i in range(N):
        sum += gaussian((x - array[i])/h)
    return sum/(N*h)
def mle(h, x, data):
    auxVar = [None] * x.shape[0]
    mle = []
    for i in range(h.shape[0]):
        for j in range(x.shape[0]):
            auxVar[j] = np.log(kde(x[j], data, h[i]))
        mle.append(np.sum(auxVar))
    return mle