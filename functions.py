import math
import numpy as np

class functions(object):
    def __init__(self):
        pass
    def __gaussian(self, x):
        return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
    def kde(self, x, array, h):
        N = len(array)
        sum = 0
        for i in range(N):
            sum += self.__gaussian((x - array[i])/h)
        return sum/(N*h)
    def mle(self, h, x, data):
        auxVar = [None] * len(x)
        mle = []
        for i in range(len(h)):
            for j in range(len(x)):
                auxVar[j] = math.log(self.kde(x[j], data, h[i]))
            mle.append(sum(auxVar))
        best = mle.index(max(mle))
        best_h = h[best]
        return best_h

    def __frac_result(self, x, data, h):
        resultFirstDer = []
        resultSecDer = []
        for i in range(len(x)):
            top = []
            bottom = []
            for j in range(len(data)):
                varAux = self.__gaussian((x[i] - data[j])/h)
                top.append(varAux*(x[i] - data[j])*(x[i] - data[j]))
                bottom.append(varAux)
            sumTop = sum(top)
            sumBottom = sum(bottom)
            resultFirstDer.append(sumTop/(sumBottom*h*h*h))
            resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
        return sum(resultFirstDer)-(len(x)/h), sum(resultSecDer)+(len(x)/(h*h))
    def nrm(self, x, data, h):
        best_h = h
        funcReturn = self.__frac_result(x, data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while abs(frac) >= 0.01:
            print(best_h)
            funcReturn = self.__frac_result(x, data, best_h)
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
        print(best_h)
        return best_h
    ######################### LOO ####################################
    def __frac_resultloo(self, data, h):
        resultFirstDer = []
        resultSecDer = []
        databkp = data.copy()
        for i in range(len(data)):
            top = []
            bottom = []
            element = databkp[i]
            databkp.pop(i)
            for j in range(len(databkp)):
                varAux = self.__gaussian((element - databkp[j])/h)
                top.append(varAux*(element - databkp[j])*(element - databkp[j]))
                bottom.append(varAux)
            databkp.clear()
            databkp = data.copy()
            sumTop = sum(top)
            sumBottom = sum(bottom)
            resultFirstDer.append(sumTop/(sumBottom*h*h*h))
            resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
        return sum(resultFirstDer)-(len(data)/h), sum(resultSecDer)+(len(data)/(h*h))
    def nrmloo(self,data, h):
        best_h = h
        funcReturn = self.__frac_resultloo(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while abs(frac) >= 0.01:
            print(best_h)
            funcReturn = self.__frac_resultloo(data, best_h)
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
        print(best_h)
        return best_h

    ######################### D DIMENSIONAL ##########################
    def __kernelNormal(self, x):
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        a = ((2*math.pi)**(-x.ndim/2))
        b = ((np.linalg.det(H))**(-0.5))
        c = np.exp(-0.5*x.transpose()*np.linalg.inv(H)*x)
        return a*b*c
    def mkde(self, x, array, h):
        N = len(array)
        sum = 0
        for i in range(N):
            sum += self.__kernelNormal((x - array[i])/h)
        return sum/(N*h**x.ndim)