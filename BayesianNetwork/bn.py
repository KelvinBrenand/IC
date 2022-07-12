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
    
    def __newtonRaphson(self, data, h=1.0, epsilon=0.01, max_iter=20):
        """Performs the Newton-Raphson's method to estimate the best value to the Kernel Density Estimation (KDE) bandwidth parameter.
           If it does not converge with the initial h value, the method will be executed again with the a new h value (h = h-1.0) until MIN_H_VALUE is reached. 

        Args:
            data (list): Datapoints to estimate from.
            h (float): Initial bandwidth value, default=1.0.
            epsilon(float): The method's threshold, defaut=0.01.
            max_iter(int): Maximum number of iterations.

        Returns:
            float: The best bandwidth value considering the input data.
            None:  If the method finds a 0 derivative or maximum number of iterations is reached. 

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

        MIN_H_VALUE = 0.4
        count = 0
        best_h = h
        funcReturn = self.__fracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while True:
            while abs(frac) >= epsilon:
                funcReturn = self.__fracResult(data, best_h)
                if funcReturn[1] == 0:
                    best_h = None
                    break
                frac = funcReturn[0]/funcReturn[1]
                best_h = best_h - frac
                count = count+1
                if count >= max_iter:
                    best_h = None
                    count = 0
                    break
            if best_h == None or best_h < 0:
                h = h-0.1
                best_h = h
                if best_h < MIN_H_VALUE:
                    print("Erro no newtonRaphson")
                    break
            else:
                break
        return best_h
    
    def __kernelDensityEstimation(self, x, data, h):
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
    
    def __multivariateNewtonRaphson(self, data, h=1.0, epsilon=0.01, max_iter=20):
        """Performs the Multivariate Newton-Raphson's method to estimate the best value to the Multivariate Kernel Density Estimation (MKDE) bandwidth parameter.
           If it does not converge with the initial h value, the method will be executed again with the a new h value (h = h-1.0) until MIN_H_VALUE is reached.

        Args:
            data (list): Datapoints to estimate from.
            h (float): Initial bandwidth value, default=1.0.
            epsilon(float): The method's threshold, defaut=0.01.
            max_iter(int): Maximum number of iterations.

        Returns:
            float: The best bandwidth value considering the input data.
            None:  If the method finds a 0 derivative or maximum number of iterations is reached.

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

        MIN_H_VALUE = 0.4
        count = 0
        best_h = h
        funcReturn = self.__multivariateFracResult(data, best_h)
        frac = funcReturn[0]/funcReturn[1]
        while True:
            while abs(frac) >= epsilon:
                funcReturn = self.__multivariateFracResult(data, best_h)
                if funcReturn[1] == 0:
                    best_h = None
                    break
                frac = funcReturn[0]/funcReturn[1]
                best_h = best_h - frac
                count = count+1
                if count >= max_iter:
                    best_h = None
                    count = 0
                    break
            if best_h == None or best_h < 0:
                h = h-0.1
                best_h = h
                if best_h < MIN_H_VALUE:
                    print("Erro no multivariateNewtonRaphson")
                    break
            else:
                break
        return best_h

    def __multivariateKernelDensityEstimation(self, x, data, h):
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

    def __LOO_Kde(self, data, h):
        """Performs the Leave-One-Out for either the 1D KDE or the Multivariate KDE.

        Args:
            data (List): Datapoints to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            list: KDE of the input data and bandwidth.
        """
        auxVar = [None] * len(data)
        databkp = data.copy()
        for i in range(len(data)):
            element = databkp[i]
            databkp.pop(i)
            if isinstance(element, float):
                auxVar[i] = self.__kernelDensityEstimation(element, databkp, h)
            else:
                auxVar[i] = self.__multivariateKernelDensityEstimation(element, databkp, h)
            databkp.clear()
            databkp = data.copy()
        return auxVar

    def __column(self, data, i):
        """Returns a specific column from a N dimensional list.

        Args:
            data (list): The complete initial list.
            i (int): The specific column index.

        Returns:
            list: The specific column of the N dimentional data.
        """
        return [row[i] for row in data]

    def __pairs(self, n):
        """Returns all possible pairs from 0 up to n. E.g.: For n = 2, it will return [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)].

        Args:
            n (int): The upper limit from where the method will generate the pairs.

        Returns:
            list of tuples: The list of all the pairs.
        """
        c = []
        for i in range(n):
            for j in range(i+1, n):
                c.append((i,j))
                c.append((j,i))
        return c
    
    def __dataPartition(self,data,index):
        """Returns two specific columns from a N dimensional list.

        Args:
            data (list): The complete initial list.
            index (tuple): The specific columns.

        Returns:
            list: The specific columns of the N dimentional data.
        """
        e = []
        for i in range(len(data)):
            f = []
            for j in index:
                f.append(data[i][j])
            e.append(f)
        return e


    def __mtxModifier(self, mtx, idx):
        """Modifies the adjacency matrix that will be returned by the MLE method.

        Args:
            mtx (list): Initial adjacency matrix.
            idx (list): The index that will be modified in the adjacency matrix.

        Returns:
            list: The modified adjacency matrix.
        """
        mtx[idx[0]][idx[1]] = 1
        return mtx

    def __trace(self, matrix):
        """Sum of diagonal elements of a matrix.

        Args:
            matrix (list): The matrix that will have its diagonal elements summed up.

        Returns:
            Int: The sum of diagonal elements.
        """
        return sum(matrix[i][i] for i in range(len(matrix[0])))
    
    def __cycle(self, matrix):
        """Detects cycles in an adjacency matrix.

        Args:
            matrix (list): The matrix that will be verified.

        Returns:
            Bool: True if the matrix has cycles. False otherwise.
        """
        An = matrix.copy()
        for i in range(1,len(matrix[0])):
            An = self.__dot(An, matrix)
            if self.__trace(An) != 0:
                return True
        return False

    def __insertedArcs(self,arcos, elem):
        """Returns a dictionary with all nodes associated to elem.

        Args:
            arcos (list): List with all the nodes and all the arcs
            elem (int): Target node

        Returns:
            dictionary: All nodes associated to elem
        """
        retorno = {}
        aux = []
        for i in arcos:
            if isinstance(i, tuple):
                if i[1] == elem:
                    aux.append(i[0])
        retorno.update({elem:aux})
        for j in aux:
            retorno.update(self.__insertedArcs(arcos,j))
        return retorno

    def __probPaths(self,arcs):#Organiza quem recebe de quem
        """All the necessary ways to calculate the probability of a node.

        Args:
            arcs (dictionary): All nodes associated to a target node

        Returns:
            list: All probability paths associated with a target node.
        """
        myDict = arcs.copy()
        myList1 = []
        for i in arcs.keys():
            if arcs.get(i) == []:
                myDict.pop(i)
        for i in myDict.keys():
            for j in myDict.get(i):
                myList1.append([i,j])
        
        if len(myList1) == 1:
            return myList1
        else:
            mylist3 = myList1.copy()
            myList2 = []
            while mylist3 != []:
                myList4 = []
                myList4.clear()
                teste = mylist3[0]
                mylist3.remove(teste)
                if mylist3 == []:
                    break
                for j in mylist3:
                    if teste[-1] == j[0]:
                        aux = teste.copy()
                        aux.append(j[1])
                        myList2.append(aux)
                        myList4.append(j)
                for m in myList4:
                    mylist3.remove(m)
                
            mylist3 = myList1.copy()
            for i in myList2:
                for j in mylist3:
                    if i[-1] == j[0]:
                        i.append(j[1])
            if myList2 == []:
                return myList1
            else:
                return myList2

    def MLE(self,data):
        """Computes the Maximum-Likelihood Estimation (MLE) of data and returns the adjacency matrix.

        Args:
            data (list): Datapoints to compute the MLE from.

        Returns:
            list: The adjacency matrix associated with data.
        """
        initial_h = 1.0
        auxVar = 1.0
        probNoIndiv = {}
        adjacency_matrix = [[0 for i in range(len(data[0]))] for n in range(len(data[0]))]
        for i in range (len(data[0])): #Inicio do código do MLE independente
            h = self.__newtonRaphson(self.__column(data, i), initial_h)
            probNoIndiv.update({(i):self.__LOO_Kde(self.__column(data, i), h)})
        auxList = []
        
        for i in range(len(probNoIndiv.get((0)))):
            for j in range(len(probNoIndiv)):
                auxVar = auxVar*probNoIndiv.get((j))[i]
            auxList.append(math.log(auxVar))
            auxVar = 1.0
        last_MLE = sum(auxList)
        print("lastMLE",last_MLE)

        arcos = list(probNoIndiv.keys()) #Inicio dos arcos
        indices = self.__pairs(len(data[0]))
        indicesCopy = indices.copy() 
        for elem in indices:
            if list(self.__insertedArcs(arcos,elem[1]).values()) == [[]]: 
                if list(self.__insertedArcs(arcos,elem[0]).values()) == [[]]: #receptor solto, emissor solto
                    myData = self.__dataPartition(data, elem)
                    h = self.__multivariateNewtonRaphson(myData, initial_h)
                    arc_Kde = self.__LOO_Kde(myData, h)

                    for i in range(len(arc_Kde)):
                        arc_Kde[i] = arc_Kde[i]/probNoIndiv.get(elem[0])[i]

                    somaDosArcosInseridos = arc_Kde.copy()
                
                else: #receptor solto, emissor já recebe
                    auxVar = self.__insertedArcs(arcos,elem[0])
                    auxVar = self.__probPaths(auxVar)
                    arcosInseridos = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        myData = self.__dataPartition(data, i)
                        h = self.__multivariateNewtonRaphson(myData, initial_h)
                        arc_Kde = self.__LOO_Kde(myData, h)

                        i.append(elem[1])
                        myData2 = self.__dataPartition(data, i)
                        h2 = self.__multivariateNewtonRaphson(myData2, initial_h)
                        arc_Kde2 = self.__LOO_Kde(myData2, h2)

                        for i in range(len(arc_Kde)):
                            arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        arcosInseridos.append(arc_Kde)

                    for i in range(len(arcosInseridos[0])):
                        somaDosArcosInseridos.append(0)
                        for j in range(len(arcosInseridos)):
                            somaDosArcosInseridos[i] = somaDosArcosInseridos[i]+arcosInseridos[j][i]
            else:
                if list(self.__insertedArcs(arcos,elem[0]).values()) == [[]]: #receptor já recebe, emissor solto
                    myData = self.__dataPartition(data, elem)
                    h = self.__multivariateNewtonRaphson(myData, initial_h)
                    arc_Kde = self.__LOO_Kde(myData, h)

                    for i in range(len(arc_Kde)):
                        arc_Kde[i] = arc_Kde[i]/probNoIndiv.get(elem[0])[i]

                    auxVar = self.__insertedArcs(arcos,elem[1])
                    auxVar = self.__probPaths(auxVar)
                    arcosJaInseridosEmNoAlvo = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        myData2 = self.__dataPartition(data, i)
                        h2 = self.__multivariateNewtonRaphson(myData2, initial_h)
                        arc_Kde2 = self.__LOO_Kde(myData2, h2)
                        
                        auxVar2 = [x for x in i if x != elem[1]]
                        myData3 = []
                        h3 = []
                        arc_Kde3 = []
                        if len(auxVar2) == 1:
                            arc_Kde3 = probNoIndiv.get(auxVar2[0])
                        else:
                            myData3 = self.__dataPartition(data, auxVar2)
                            h3 = self.__multivariateNewtonRaphson(myData3, initial_h)
                            arc_Kde3 = self.__LOO_Kde(myData3, h3)
                        
                        for i in range(len(arc_Kde)):
                            arc_Kde2[i] = arc_Kde2[i]/arc_Kde3[i]
                        
                        arcosJaInseridosEmNoAlvo.append(arc_Kde2)
                    
                    for i in range(len(arcosJaInseridosEmNoAlvo[0])):
                        somaDosArcosInseridos.append(0)
                        for j in range(len(arcosJaInseridosEmNoAlvo)):
                            somaDosArcosInseridos[i] = somaDosArcosInseridos[i]+arcosJaInseridosEmNoAlvo[j][i]+arc_Kde[i]

                else: #receptor já recebe, emissor já recebe
                    auxVar = self.__insertedArcs(arcos,elem[0])
                    auxVar = self.__probPaths(auxVar)
                    arcosInseridos = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        myData = self.__dataPartition(data, i)
                        h = self.__multivariateNewtonRaphson(myData, initial_h)
                        arc_Kde = self.__LOO_Kde(myData, h)

                        i.append(elem[1])
                        myData2 = self.__dataPartition(data, i)
                        h2 = self.__multivariateNewtonRaphson(myData2, initial_h)
                        arc_Kde2 = self.__LOO_Kde(myData2, h2)

                        for i in range(len(arc_Kde)):
                            arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        arcosInseridos.append(arc_Kde)

                    auxVar = self.__insertedArcs(arcos,elem[1])
                    auxVar = self.__probPaths(auxVar)
                    for i in auxVar:
                        myData2 = self.__dataPartition(data, i)
                        h2 = self.__multivariateNewtonRaphson(myData2, initial_h)
                        arc_Kde2 = self.__LOO_Kde(myData2, h2)
                        
                        auxVar2 = [x for x in i if x != elem[1]]
                        myData3 = []
                        h3 = []
                        arc_Kde3 = []
                        if len(auxVar2) == 1:
                            arc_Kde3 = probNoIndiv.get(auxVar2[0])
                        else:
                            myData3 = self.__dataPartition(data, auxVar2)
                            h3 = self.__multivariateNewtonRaphson(myData3, initial_h)
                            arc_Kde3 = self.__LOO_Kde(myData3, h3)
                        
                        for i in range(len(arc_Kde)):
                            arc_Kde2[i] = arc_Kde2[i]/arc_Kde3[i]
                        
                        arcosInseridos.append(arc_Kde2)
                    
                    for i in range(len(arcosInseridos[0])):
                        somaDosArcosInseridos.append(0)
                        for j in range(len(arcosInseridos)):
                            somaDosArcosInseridos[i] = somaDosArcosInseridos[i]+arcosInseridos[j][i]
            
            try:
                indicesCopy.index((elem[1],elem[0]))
                indicePar0 = elem
                resultadoPar0 = somaDosArcosInseridos.copy()
                indicesCopy.remove(elem)
            except:
                indicePar1 = elem
                resultadoPar1 = somaDosArcosInseridos.copy()
                mtxCopy = copy.deepcopy(adjacency_matrix)
                mtxCopy = self.__mtxModifier(mtxCopy,indicePar0)
                mlePar0 = None
                if not self.__cycle(mtxCopy):
                    auxDictPar0 = probNoIndiv.copy()
                    auxDictPar0.update({indicePar0[1]:resultadoPar0})
                    mlePar0 = []
                    for i in range(len(auxDictPar0.get(list(auxDictPar0.keys())[0]))):
                        mlePar0.append(1)
                        for j in range(len(auxDictPar0.keys())):
                            mlePar0[i] = mlePar0[i]*auxDictPar0.get(j)[i]
                        mlePar0[i] = math.log(mlePar0[i])
                    mlePar0 = sum(mlePar0)
                
                mtxCopy = copy.deepcopy(adjacency_matrix)
                mtxCopy = self.__mtxModifier(mtxCopy,indicePar1)
                mlePar1 = None
                if not self.__cycle(mtxCopy):
                    auxDictPar1 = probNoIndiv.copy()
                    auxDictPar1.update({indicePar1[1]:resultadoPar1})
                    mlePar1 = []
                    for i in range(len(auxDictPar1.get(list(auxDictPar1.keys())[0]))):
                        mlePar1.append(1)
                        for j in range(len(auxDictPar1.keys())):
                            mlePar1[i] = mlePar1[i]*auxDictPar1.get(j)[i]
                        mlePar1[i] = math.log(mlePar1[i])
                    mlePar1 = sum(mlePar1)

                if mlePar0 != None and mlePar1 != None:
                    print("mlePar0",mlePar0)
                    print("mlePar1",mlePar1)
                    if mlePar0 > mlePar1:
                        if mlePar0 > last_MLE:
                            last_MLE = mlePar0
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, indicePar0)
                            probNoIndiv = auxDictPar0.copy()
                            arcos.append(indicePar0) 
                    else:
                        if mlePar1 > last_MLE:
                            last_MLE = mlePar1
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, indicePar1)
                            probNoIndiv = auxDictPar1.copy()
                            arcos.append(indicePar1)
                    print("lastMLE", last_MLE)

        return adjacency_matrix