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

def nrm_numerator(x, data, h):
    result = []
    for i in range(x.shape[0]):
        top = []
        bottom = []
        for j in range(len(data)):
            varAux = gaussian((x[i] - data[j])/h)
            bottom.append(varAux)
            top.append(varAux*(x[i] - data[j])*(x[i] - data[j]))
        result.append(sum(top)/(sum(bottom)*h*h*h))
    return sum(result)-(x.shape[0]/h)
def nrm_denominator(x, data, h):
    result = []
    for i in range(x.shape[0]):
        top = []
        bottom = []
        for j in range(len(data)):
            varAux = gaussian((x[i] - data[j])/h)
            bottom.append(varAux)
            top.append(varAux*(x[i] - data[j])*(x[i] - data[j]))
        result.append((sum(top)*-3)/(sum(bottom)*h*h*h*h))
    return sum(result)-(x.shape[0]/(h*h))
def nrm(x, data, h):
    best_h = h
    frac = nrm_numerator(x, data, best_h)/nrm_denominator(x, data, best_h)
    while abs(frac) >= 0.0001:
        frac = nrm_numerator(x, data, best_h)/nrm_denominator(x, data, best_h)
        best_h = best_h - frac
    return best_h