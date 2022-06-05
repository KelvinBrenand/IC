# Author: Kelvin Brenand <brenand.kelvin@gmail.com>

import math
import copy

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
        if isinstance(A, list) and isinstance(B, list) and len(A) == len(B):
            if isinstance(A[0], list) and isinstance(B[0], list) and len(A[0]) == len(B[0]):
                result = [[0 for i in range(len(A[0]))] for n in range(len(A[0]))]
                for i in range(len(A)):
                    for j in range(len(B[0])):
                        for k in range(len(B)):
                            result[i][j] += A[i][k] * B[k][j]
                return result
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

    def LOO_Kde(self, data, h): #virou uma função de fazer Leave-One-Out, basicamente.
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

    def __column(self, matrix, i):
        return [row[i] for row in matrix]

    def __pairs(self, n):
        c = []
        for i in range(n):
            for j in range(i+1, n):
                c.append((i,j))
        return c
    
    def __list2a2(self, data,index):
        e = []
        for i in range(len(data)):
            e.append([data[i][index[0]],data[i][index[1]]])
        return e


    def __mtxModifier(self, mtx, idx):
        mtx[idx[0]][idx[1]] = 1
        return mtx

    def __trace(self, matrix):
        return sum(matrix[i][i] for i in range(len(matrix[0])))
    
    def __cycle(self, matrix):
        An = matrix
        for i in range(1,len(matrix[0])):
            An = self.__dot(An, matrix)
            if self.__trace(An) != 0:
                return True
        return False

    def MLE(self,data):
        MIN = 0.4
        initial_h = 1.0
        auxVar = 1.0
        probs = {}
        initial_adjacency_matrix = [[0 for i in range(len(data[0]))] for n in range(len(data[0]))]
        adjacency_matrix = []
        for i in range (len(data[0])):
            while(True):
                h = self.newtonRaphson(self.__column(data, i), initial_h)
                initial_h -= 0.1
                if not isinstance(h, str) and h != None and h > 0:
                    initial_h = 1.0
                    break
                if initial_h <= MIN:
                    print("NewtonRaphson não convergiu")
                    return None
            probs.update({(i):self.LOO_Kde(self.__column(data, i), h)})

        auxList = []
        for i in range(len(probs.get((0)))):
            for j in range(len(probs)):
                auxVar = auxVar*probs.get((j))[i]
            auxList.append(math.log(auxVar))
            auxVar = 1.0
        last_MLE = sum(auxList)

        indices = self.__pairs(len(data[0]))
        for elem in indices:
            myList2a2 = self.__list2a2(data, elem)
            while(True):
                h = self.multivariateNewtonRaphson(myList2a2, initial_h)
                initial_h -= 0.1
                if not isinstance(h, str) and h != None and h > 0:
                    initial_h = 1.0
                    break
                if initial_h <= MIN:
                    print("MultivariateNewtonRaphson não convergiu")
                    return None
            probs.update({elem:self.LOO_Kde(myList2a2, h)})

            arcIdx = []
            auxList2 = []
            auxList3 = []
            for i in range(len(probs.get(elem))):
                auxVar2 = probs.get(elem)[i]/probs.get((elem[0]))[i]
                auxVar3 = probs.get(elem)[i]/probs.get((elem[1]))[i]
                auxVarRange = [x for x in range(len(data[0]))]
                auxVarRange.remove(elem[1])
                for j in auxVarRange:
                    auxVar2 = auxVar2*probs.get((j))[i]
                auxList2.append(math.log(auxVar2))
                auxVarRange = [x for x in range(len(data[0]))]
                auxVarRange.remove(elem[0])
                for j in auxVarRange:
                    auxVar3 = auxVar3*probs.get((j))[i]
                auxList3.append(math.log(auxVar3))
            auxVar4 = sum(auxList2)
            auxVar5 = sum(auxList3)
            if auxVar4 > last_MLE or auxVar5 > last_MLE:
                adjacency_matrix = copy.deepcopy(initial_adjacency_matrix)
                if auxVar4 > auxVar5:
                    arcIdx = [elem[1],elem[0]]
                    last_MLE = auxVar4
                else:
                    arcIdx = list(elem)
                    last_MLE = auxVar5
                adjacency_matrix = self.__mtxModifier(adjacency_matrix,arcIdx)
        
        # arcIdx.sort()
        # indices.remove(tuple(arcIdx))
        # for elem in indices: #TODO adicionar proximos arcos, dado o primeiro
        #     #if elem[0] == arcIdx
        #     pass
        return adjacency_matrix