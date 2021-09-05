import math
import numpy as np

class functions(object):
    def __init__(self):
        pass
    def __gaussian(self, x):
        return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
    def kernelDensityEstimation(self, data, h):
        #Leave one out
        kde_result = []
        databkp = data.copy()
        for i in range(len(data)):
            element = databkp[i]
            databkp.pop(i)
            sum = 0
            for i in range(len(data)):
                sum += self.__gaussian((element - data[i])/h)
            kde_result.append(sum/(len(data)*h))
            databkp.clear()
            databkp = data.copy()
        return kde_result
    def maximumLikelihoodEstimation(self, h, x, data):
        auxVar = [None] * len(x)
        mle = []
        for i in range(len(h)):
            for j in range(len(x)):
                auxVar[j] = math.log(self.kde(x[j], data, h[i]))
            mle.append(sum(auxVar))
        best = mle.index(max(mle))
        best_h = h[best]
        return best_h
    def __fracResult(self, data, h):
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
    def newtonRaphson(self, data, h):
        #Leave one out
        best_h = h
        funcReturn = self.__fracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while abs(frac) >= 0.01:
            print(best_h)
            funcReturn = self.__fracResult(data, best_h)
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
        print(best_h)
        return best_h
    def __multidimensionalGaussian(self, x):
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        a = ((2*math.pi)**(-x.ndim/2))
        b = ((np.linalg.det(H))**(-0.5))
        c = np.exp(-0.5*x.transpose()*np.linalg.inv(H)*x)
        return a*b*c
    def multidimensionalKernelDensityEstimation(self, data, h):
        #Leave one out
        kde_result = []
        databkp = data
        for i in range(len(data)):
            element = databkp[i]
            databkp = np.delete(databkp, i, 0)
            sum = 0
            for i in range(len(data)):
                sum += self.__multidimensionalGaussian((element - data[i])/h)
            kde_result.append(sum/(len(data)*h**element.ndim))
            databkp = data
        return kde_result
    #################################################################
    def __multidimensionalFracResult(self, data, h):
        H = np.array([[1.0, 0.0],[0.0, 1.0]])
        resultFirstDer = []
        resultSecDer = []
        databkp = data
        for i in range(len(data)):
            top = []
            bottom = []
            element = databkp[i]
            databkp = np.delete(databkp, i, 0)
            for j in range(len(databkp)):
                varAux = self.__multidimensionalGaussian((element - databkp[j])/h)
                top.append(varAux*(1/(h*h*h))*(element - databkp[j]).transpose()*np.linalg.inv(H)*(element - databkp[j]))
                bottom.append(varAux)
            databkp = data
            sumTop = sum(top)
            sumBottom = sum(bottom)
            resultFirstDer.append(sumTop/sumBottom)
            resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
        return sum(resultFirstDer)-(len(data)*data.ndim/h), sum(resultSecDer)+(len(data)*data.ndim/(h*h))
    def multivariateNewtonRaphson(self, data, h):
        #Leave one out
        best_h = h
        funcReturn = self.__multidimensionalFracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        print(funcReturn)
        while abs(frac) >= 0.01:
            print(best_h)
            funcReturn = self.__multidimensionalFracResult(data, best_h)
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
        print(best_h)
        return best_h