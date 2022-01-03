import math

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

    def __identityMatrix(self, n):
        m=[[0. for x in range(n)] for y in range(n)]
        for i in range(0,n):
            m[i][i] = 1.
        return m
    def __ndim(self, x):
        if isinstance(x[0], float):
            return 1
        return len(x[0])
    def __listSubtraction(self, x, y):
        result = []
        for i in range(len(x)):
            result.append(x[i] - y[i])
        return result
    def __listDivision(self, x, y):
        result = []
        for i in range(len(x)):
            result.append(x[i]/y)
        return result
    def __dot(self, A,B):
        try:
            auxVar = len(B[0])
            result = [0.] * len(A)
            for i in range(len(A)):
                for j in range(len(B)):
                    result[i] += A[j] * B[j][i]
        except:
            result = 0.
            for i in range(len(A)):
                result += A[i]*B[i]
        return result
    def __multidimensionalGaussian(self, x):
        H = self.__identityMatrix(len(x))
        idenMatrixDet = 1.0
        idenMatrixInv = H
        a = ((2*math.pi)**(-self.__ndim(x)/2))
        b = ((idenMatrixDet)**(-0.5))
        c = math.exp(-0.5*self.__dot(self.__dot(x, idenMatrixInv), x)) #x.T
        return a*b*c
    def multidimensionalKernelDensityEstimation(self, data, h):
        #Leave one out
        kde_result = []
        data = data.tolist()
        databkp = data.copy()
        for i in range(len(data)):
            element = databkp[i]
            databkp.pop(i)
            sum = 0
            for i in range(len(data)):
                sum += self.__multidimensionalGaussian(self.__listDivision(self.__listSubtraction(element, data[i]), h))
            kde_result.append(sum/(len(data)*h**self.__ndim(element)))
            databkp = data.copy()
        return kde_result
    def __multidimensionalFracResult(self, data, h):
        H = self.__identityMatrix(self.__ndim(data))
        idenMatrixInv = H
        resultFirstDer = []
        resultSecDer = []
        data = data.tolist()
        databkp = data.copy()
        for i in range(len(data)):
            top = []
            bottom = []
            element = databkp[i]
            databkp.pop(i)
            for j in range(len(databkp)):
                varAux = self.__multidimensionalGaussian(self.__listDivision(self.__listSubtraction(element, databkp[j]), h))
                top.append(varAux*(1/(h*h*h))*self.__dot(self.__dot(self.__listSubtraction(element, databkp[j]), idenMatrixInv), self.__listSubtraction(element, databkp[j])))
                bottom.append(varAux)
            databkp = data.copy()
            sumTop = sum(top)
            sumBottom = sum(bottom)
            resultFirstDer.append(sumTop/sumBottom)
            resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
        return sum(resultFirstDer)-(len(data)*self.__ndim(data)/h), sum(resultSecDer)+(len(data)*self.__ndim(data)/(h*h))
    def multidimensionalNewtonRaphson(self, data, h):
        #Leave one out
        best_h = h
        funcReturn = self.__multidimensionalFracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while abs(frac) >= 0.01:
            print(best_h)
            funcReturn = self.__multidimensionalFracResult(data, best_h)
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
        print(best_h)
        return best_h