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

def frac_result(x, data, h):
    resultFirstDer = []
    resultSecDer = []
    for i in range(x.shape[0]):
        top = []
        bottom = []
        for j in range(len(data)):
            varAux = gaussian((x[i] - data[j])/h)
            top.append(varAux*(x[i] - data[j])*(x[i] - data[j]))
            bottom.append(varAux)
        sumTop = sum(top)
        sumBottom = sum(bottom)
        resultFirstDer.append(sumTop/(sumBottom*h*h*h))
        resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
    return sum(resultFirstDer)-(x.shape[0]/h), sum(resultSecDer)-(x.shape[0]/(h*h))
def nrm(x, data, h):
    best_h = h
    funcReturn = frac_result(x, data, best_h)
    frac = funcReturn[0]/funcReturn[1]
    while abs(frac) >= 0.001:
        funcReturn = frac_result(x, data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        best_h = best_h - frac
    return best_h