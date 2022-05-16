# Author: Kelvin Brenand <brenand.kelvin@gmail.com>

import math

class newtonRapson(object):
    '''
    This class implements the 1D and nD Newton-Raphson's bandwidth estimator method, its helping methods, and the 1D and nD Kernel Density Estimation method.
    '''
    def __gaussian(self, x):
        """Computes the gaussian kernel of a given value.

        Args:
            x (float): Value to be computed.

        Returns:
            float: Gaussian kernel of the input value.
        """
        return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))

    def __fracResult(self, data, h):
        """Function that performs the Newton-Raphson's method calculations using the Leave-One-Out technique.

        Args:
            data (list): Datapoints to estimate from.
            h (float): Bandwidth parameter.

        Returns:
            tuple: Tuple consisting of the numerator and denominator of the fraction present in Newton-Raphson's method.
        """
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
    
    def newtonRaphson(self, data, h=1.0, epsilon=0.01, max_iter=20):
        """Performs the Newton-Raphson's method to estimate the best value to the Kernel Density Estimation (KDE) bandwidth parameter.

        Args:
            data (list): Datapoints to estimate from.
            h (float): Initial bandwidth value, default=1.0
            epsilon(float): The method's threshold, defaut=0.01
            max_iter(int): Maximum number of iterations

        Returns:
            float: The best bandwidth value considering the input data.
            None:  If the method finds a 0 derivative or maximum number of iterations is reached 

        Error Messages:
            EXIT_FAIL_INVALID_DATA_TYPE: Input set type must be a list.
            EXIT_FAIL_INVALID_H_TYPE: The h parameter must be a float.
            EXIT_FAIL_INVALID_EPSILON_TYPE: The epsilon parameter must be a float.
            EXIT_FAIL_INVALID_H: The value of the parameter h must be greater than zero.
            EXIT_FAIL_INVALID_EPSILON: The value of the parameter epsilon must be greater than zero.
        """
        
        if not type(data) is list:
            return "EXIT_FAIL_INVALID_DATA_TYPE"

        if not type(h) is float:
            return "EXIT_FAIL_INVALID_H_TYPE"

        if not type(epsilon) is float:
            return "EXIT_FAIL_INVALID_EPSILON_TYPE"

        if h <= 0.0:
            return "EXIT_FAIL_INVALID_H"
            
        if epsilon <= 0.0:
            return "EXIT_FAIL_INVALID_EPISILON"

        best_h = h
        funcReturn = self.__fracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        count = 0
        while abs(frac) >= epsilon:
            funcReturn = self.__fracResult(data, best_h)
            if funcReturn[1] == 0:
                return None
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
            count = count+1
            if count >= max_iter:
                return None
        return best_h
    
    def kernelDensityEstimation(self, x, data, h):
        """Computes the Kernel Density Estimation (KDE) using the gaussian kernel and the Leave-One-Out technique of the given datapoints and bandwidth parameter.

        Args:
            x (float): Point where the kde will be estimated.
            data (list): Datapoints to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            list: KDE of the input data and bandwidth.

        Error Messages:
            EXIT_FAIL_INVALID_DATA_TYPE: Input set type must be a list.
            EXIT_FAIL_INVALID_H_TYPE: The h parameter must be a float.
            EXIT_FAIL_INVALID_H: The value of the parameter h must be greater than zero.
        """
        
        if not type(data) is list:
            return "EXIT_FAIL_INVALID_DATA_TYPE"

        if not type(h) is float:
            return "EXIT_FAIL_INVALID_H_TYPE"

        if h <= 0.0:
            return "EXIT_FAIL_INVALID_H"
    
        sum = 0
        for i in range(len(data)):
            sum += self.__gaussian((x - data[i])/h)
        return (sum/(len(data)*h))

    ###################### MULTIDIMENTIONAL NEWTON RAPHSON ######################

    def __identityMatrix(self, n):
        """Generates an identity matrix of dimension NxN.

        Args:
            n (int): Matrix's dimension.

        Returns:
            list: Identity matrix.
        """
        m=[[0. for x in range(n)] for y in range(n)]
        for i in range(0,n):
            m[i][i] = 1.
        return m

    def __ndim(self, x):
        """Number of array dimensions.

        Args:
            x (list): value that should have its number of dimensions found.

        Returns:
            int: Number of dimensions.
        """
        if isinstance(x[0], float):
            return 1
        return len(x[0])

    def __listSubtraction(self, x, y):
        """Performs the difference between two given lists.

        Args:
            x (list): First list.
            y (list): Second list.

        Returns:
            list: The difference between the two lists.
        """
        result = []
        for i in range(len(x)):
            result.append(x[i] - y[i])
        return result

    def __listDivision(self, x, y):
        """Performs the division operation between the elements of two given lists.

        Args:
            x (list): First list.
            y (list): Second list.

        Returns:
            list: The result of the division.
        """
        result = []
        for i in range(len(x)):
            result.append(x[i]/y)
        return result

    def __dot(self, A,B):
        """Dot product of two values.

        Args:
            A (list): First list.
            B (list): Second list.

        Returns:
            list or float: If A and B are both 1D, it returns the resulting float of the inner product. If B is 2D, it returns the resulting list of the sum product.
        """
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

    def __multivariateGaussian(self, x):
        """Computes the multivariate gaussian kernel of a given value. 

        Args:
            x (list): Value to be computed.

        Returns:
            float: Multivariate gaussian kernel of the input value.
        """
        H = self.__identityMatrix(len(x))
        idenMatrixDet = 1.0
        idenMatrixInv = H
        return ((2*math.pi)**(-self.__ndim(x)/2))*((idenMatrixDet)**(-0.5))*(math.exp(-0.5*self.__dot(self.__dot(x, idenMatrixInv), x)))
    
    def __multivariateFracResult(self, data, h):
        """Function that performs the Mutivariate Newton-Raphson's method calculations using the Leave-One-Out technique.

        Args:
            data (list): Datapoints to estimate from.
            h (float): Bandwidth parameter.

        Returns:
            tuple: Tuple consisting of the numerator and denominator of the fraction present in Newton-Raphson's method.
        """
        H = self.__identityMatrix(self.__ndim(data))
        idenMatrixInv = H
        resultFirstDer = []
        resultSecDer = []
        databkp = data.copy()
        for i in range(len(data)):
            top = []
            bottom = []
            element = databkp[i]
            databkp.pop(i)
            for j in range(len(databkp)):
                varAux = self.__multivariateGaussian(self.__listDivision(self.__listSubtraction(element, databkp[j]), h))
                top.append(varAux*(1/(h*h*h))*self.__dot(self.__dot(self.__listSubtraction(element, databkp[j]), idenMatrixInv), self.__listSubtraction(element, databkp[j])))
                bottom.append(varAux)
            databkp = data.copy()
            sumTop = sum(top)
            sumBottom = sum(bottom)
            resultFirstDer.append(sumTop/sumBottom)
            resultSecDer.append((sumTop*-3)/(sumBottom*h*h*h*h))
        return sum(resultFirstDer)-(len(data)*self.__ndim(data)/h), sum(resultSecDer)+(len(data)*self.__ndim(data)/(h*h))
    
    def multivariateNewtonRaphson(self, data, h=1.0, epsilon=0.01, max_iter=20):
        """Performs the Multivariate Newton-Raphson's method to estimate the best value to the Multivariate Kernel Density Estimation (MKDE) bandwidth parameter.

        Args:
            data (list): Datapoints to estimate from.
            h (float): Initial bandwidth value, default=1.0
            epsilon(float): The method's threshold, defaut=0.01
            max_iter(int): Maximum number of iterations

        Returns:
            float: The best bandwidth value considering the input data.
            None:  If the method finds a 0 derivative or maximum number of iterations is reached

        Error Messages:
            EXIT_FAIL_INVALID_DATA_TYPE: Input set type must be a list.
            EXIT_FAIL_INVALID_H_TYPE: The h parameter must be a float.
            EXIT_FAIL_INVALID_EPSILON_TYPE: The epsilon parameter must be a float.
            EXIT_FAIL_INVALID_H: The value of the parameter h must be greater than zero.
            EXIT_FAIL_INVALID_EPSILON: The value of the parameter epsilon must be greater than zero.
        """
        
        if not type(data) is list:
            return "EXIT_FAIL_INVALID_DATA_TYPE"

        if not type(h) is float:
            return "EXIT_FAIL_INVALID_H_TYPE"

        if not type(epsilon) is float:
            return "EXIT_FAIL_INVALID_EPSILON_TYPE"

        if h <= 0.0:
            return "EXIT_FAIL_INVALID_H"
            
        if epsilon <= 0.0:
            return "EXIT_FAIL_INVALID_EPISILON"

        best_h = h
        funcReturn = self.__multivariateFracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        count = 0
        while abs(frac) >= epsilon:
            funcReturn = self.__multivariateFracResult(data, best_h)
            if funcReturn[1] == 0:
                return None
            frac = funcReturn[0]/funcReturn[1]
            best_h = best_h - frac
            count = count+1
            if count >= max_iter:
                return None
        return best_h

    def multivariateKernelDensityEstimation(self, x, data, h):
        """Computes the Multivariate Kernel Density Estimation (MKDE) using the multivariate gaussian kernel and the Leave-One-Out technique of the given datapoints and bandwidth parameter.

        Args:
            x (list): N dimentional point where the mKDE will be estimated.
            data (list): Datapoints to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            list: KDE of the input data and bandwidth.

        Error Messages:
            EXIT_FAIL_INVALID_DATA_TYPE: Input set type must be a list.
            EXIT_FAIL_INVALID_H_TYPE: The h parameter must be a float.
            EXIT_FAIL_INVALID_H: The value of the parameter h must be greater than zero.
        """
        
        if not type(data) is list:
            return "EXIT_FAIL_INVALID_DATA_TYPE"

        if not type(h) is float:
            return "EXIT_FAIL_INVALID_H_TYPE"

        if h <= 0.0:
            return "EXIT_FAIL_INVALID_H"
            
        sum = 0
        for i in range(len(data)):
            sum += self.__multivariateGaussian(self.__listDivision(self.__listSubtraction(x, data[i]), h))
        return (sum/(len(data)*h**self.__ndim(x)))

    def maximumLikelihoodEstimation(self, data, h): #virou uma função de fazer Leave-One-Out, basicamente.
        auxVar = [None] * len(data)
        databkp = data.copy()
        for i in range(len(data)):
            element = databkp[i]
            databkp.pop(i)
            if isinstance(element, float):
                auxVar[i] = self.kernelDensityEstimation(element, databkp, h)
            else:
                auxVar[i] = self.multivariateKernelDensityEstimation(element, databkp, h)
            databkp.clear()
            databkp = data.copy()
        return auxVar