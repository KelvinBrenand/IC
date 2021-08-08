import math

def gaussian(x):
    return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
def kde(x, array, h):
    N = len(array)
    sum = 0
    for i in range(N):
        sum += gaussian((x - array[i])/h)
    return sum/(N*h)
def mle(h, x, data):
    auxVar = [None] * len(x)
    mle = []
    for i in range(len(h)):
        for j in range(len(x)):
            auxVar[j] = math.log(kde(x[j], data, h[i]))
        mle.append(sum(auxVar))
    best = mle.index(max(mle))
    best_h = h[best]
    return best_h

def frac_result(x, data, h):
    resultFirstDer = []
    resultSecDer = []
    for i in range(len(x)):
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
    return sum(resultFirstDer)-(len(x)/h), sum(resultSecDer)-(len(x)/(h*h))
def nrm(x, data, h):
    best_h = h
    funcReturn = frac_result(x, data, best_h)
    frac = funcReturn[0]/funcReturn[1]
    while abs(frac) >= 0.01:
        print(best_h)
        funcReturn = frac_result(x, data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        best_h = best_h - frac
    print(best_h)
    return best_h