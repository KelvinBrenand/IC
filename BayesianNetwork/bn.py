# Author: Kelvin Brenand <brenand.kelvin@gmail.com>

import math
import copy
import random
import pickle as pkl

class BayesianNetwork(object):
    '''
    This class implements the 1D and nD Newton-Raphson's bandwidth estimator method, its helping methods, and the 1D and nD Kernel Density Estimation method.
    '''
    
    def __init__(self, data):
        self.data = data
        self._graphProbabilities = {}
        self._adjacencyMatrix = []

    def __gaussian(self, x):
        """Computes the gaussian kernel of a given value.

        Args:
            x (float): Value to be computed.

        Returns:
            float: Gaussian kernel of the input value.
        """
        return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
    
    def __kernelDensityEstimation(self, x, data, h):
        """Computes the Kernel Density Estimation (KDE) using the gaussian kernel and the Leave-One-Out technique of the given datapoints and bandwidth parameter.

        Args:
            x (float): Point where the kde will be estimated.
            data (list): Datapoints to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            list: KDE of the input data and bandwidth.
        """
    
        sum = 0
        for i in range(len(data)):
            sum += self.__gaussian((x - data[i])/h)
        return (sum/(len(data)*h))

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
    
    def __multivariateKernelDensityEstimation(self, x, data, h):
        """Computes the Multivariate Kernel Density Estimation (MKDE) using the multivariate gaussian kernel and the Leave-One-Out technique of the given datapoints and bandwidth parameter.

        Args:
            x (list): N dimentional point where the mKDE will be estimated.
            data (list): Datapoints to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            list: KDE of the input data and bandwidth.
        """

        sum = 0
        for i in range(len(data)):
            sum += self.__multivariateGaussian(self.__listDivision(self.__listSubtraction(x, data[i]), h))
        return (sum/(len(data)*h**self.__ndim(x)))

    def __intervalH(self, data):
        """Computes the right endpoint of the interval of values h can assume. The other endpoint is 0.1.

        Args:
            data (List): Datapoints to compute the interval from.

        Returns:
            float: The right endpoint.
        """
        menDistPontos = []
        for i in data:
            distPonto = []
            for j in data:
                if i == j: continue
                if isinstance(i, float):
                    distPonto.append(math.dist([i],[j]))
                else:
                    distPonto.append(math.dist(i,j))
            menDistPontos.append(min(distPonto))
        return max(menDistPontos)

    def __LOO_Kde(self, data, index, num_h):
        """Performs the bandwidth estimation and the Leave-One-Out KDE for either the 1D KDE or the Multivariate KDE.

        Args:
            data (List): Datapoints to compute the KDE from.
            index (tuple/int): The index of comlumns to use in the data partition method.
            num_h (int): The amount of h to be used to compute the best KDE. 

        Returns:
            list: KDE of myData and bandwidth h.
        """
        myData = self.__dataPartition(data, index)
        rightEndpoint = self.__intervalH(myData)
        hs = {round(random.uniform(0.1, rightEndpoint),1) for x in range(num_h)}
        logsAndKdes = {}
        for h in hs:
            kde = []
            databkp = myData.copy()
            for i in range(len(myData)):
                element = databkp[i]
                databkp.pop(i)
                if isinstance(element, float):
                    kde.append(self.__kernelDensityEstimation(element, databkp, h))
                else:
                    kde.append(self.__multivariateKernelDensityEstimation(element, databkp, h))
                databkp.clear()
                databkp = myData.copy()
            log = []
            for i in range(len(kde)):
                log.append(math.log(kde[i]))
            logsAndKdes.update({sum(log):kde})
        return logsAndKdes.get(max(logsAndKdes))

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
        """Returns n specific columns from a N dimensional list.

        Args:
            data (list): The complete initial list.
            index (tuple/int): The n specific columns.

        Returns:
            list: The specific columns of the N dimentional data.
        """
        if isinstance(index, int):
            return [row[index] for row in data]
        else:
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
            arcos (list): List with all the nodes and all the arcs.
            elem (int): Target node

        Returns:
            dictionary: All nodes associated to elem.
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

    def __probPaths(self,arcs):
        """All the necessary ways to calculate the probability of a node.

        Args:
            arcs (dictionary): All nodes associated to a target node.

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
        myList2 = []
        for i in myList1:
            flag = True
            for j in myList1:
                if i[-1] == j[0]:
                    flag = False
                    aux = i.copy()
                    aux.append(j[1])
                    myList2.append(aux)
            if flag:
                    myList2.append(i)
        myList3 = myList2.copy()
        for i in myList2:
            for j in myList2:
                if i[1] == j[0]:
                    try:
                        myList3.remove(j)
                    except:
                        pass
        for i in myList3:
            for j in myList1:
                if i[-1] == j[0]:
                    aux = i.copy()
                    aux.append(j[1])
                    try:
                        myList3.remove(i)
                    except:
                        pass
                    myList3.append(aux)
        return myList3

    def fit(self, num_h=100):
        """Computes the Maximum-Likelihood Estimation (MLE) of data and returns the adjacency matrix.

        Args:
            data (list): Datapoints to compute the MLE from.
            num_h (int): The amount of h to be used to compute the best KDE. 

        Raises:
            TypeError: data must be list.

        Returns:
            list: The adjacency matrix associated with data.
            None: If a RuntimeError happend.
        """

        if not type(self.data) is list:
            raise TypeError("data must be list")

        auxVar = 1.0
        probs = {}
        adjacency_matrix = [[0 for i in range(len(self.data[0]))] for n in range(len(self.data[0]))]
        for i in range (len(self.data[0])):
            nodeKde = self.__LOO_Kde(self.data, i, num_h)
            if nodeKde == None: return None
            probs.update({(i):nodeKde})
        auxList = []
        
        for i in range(len(probs.get((0)))):
            for j in range(len(probs)):
                auxVar = auxVar*probs.get((j))[i]
                if auxVar == 0:
                    auxVar = 0.001
            auxList.append(math.log(auxVar))
            auxVar = 1.0
        last_MLE = sum(auxList)

        arcos = list(probs.keys())
        indices = self.__pairs(len(self.data[0]))
        indicesCopy = indices.copy() 
        for elem in indices:
            if list(self.__insertedArcs(arcos,elem[1]).values()) == [[]]: 
                if list(self.__insertedArcs(arcos,elem[0]).values()) == [[]]:
                    arc_Kde = self.__LOO_Kde(self.data, elem, num_h)
                    if arc_Kde == None: return None
                    for i in range(len(arc_Kde)):
                        denom = probs.get(elem[0])[i]
                        if denom == 0:
                            denom = 0.001
                        arc_Kde[i] = arc_Kde[i]/denom
                    somaDosArcosInseridos = arc_Kde.copy()
                
                else:
                    auxVar = self.__insertedArcs(arcos,elem[0])
                    auxVar = self.__probPaths(auxVar)
                    arcosInseridos = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        arc_Kde = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde == None: return None
                        i.append(elem[1])
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        for i in range(len(arc_Kde)):
                            arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        arcosInseridos.append(arc_Kde)

                    for i in range(len(arcosInseridos[0])):
                        somaDosArcosInseridos.append(0)
                        for j in range(len(arcosInseridos)):
                            somaDosArcosInseridos[i] = somaDosArcosInseridos[i]+arcosInseridos[j][i]
            else:
                if list(self.__insertedArcs(arcos,elem[0]).values()) == [[]]:
                    arc_Kde = self.__LOO_Kde(self.data, elem, num_h)
                    if arc_Kde == None: return None
                    for i in range(len(arc_Kde)):
                        arc_Kde[i] = arc_Kde[i]/probs.get(elem[0])[i]

                    auxVar = self.__insertedArcs(arcos,elem[1])
                    auxVar = self.__probPaths(auxVar)
                    arcosJaInseridosEmNoAlvo = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        auxVar2 = [x for x in i if x != elem[1]]
                        arc_Kde3 = []
                        if len(auxVar2) == 1:
                            arc_Kde3 = probs.get(auxVar2[0])
                        else:
                            arc_Kde3 = self.__LOO_Kde(self.data, auxVar2, num_h)
                            if arc_Kde3 == None: return None
                        for i in range(len(arc_Kde)):
                            arc_Kde2[i] = arc_Kde2[i]/arc_Kde3[i]
                        
                        arcosJaInseridosEmNoAlvo.append(arc_Kde2)
                    
                    for i in range(len(arcosJaInseridosEmNoAlvo[0])):
                        somaDosArcosInseridos.append(0)
                        for j in range(len(arcosJaInseridosEmNoAlvo)):
                            somaDosArcosInseridos[i] = somaDosArcosInseridos[i]+arcosJaInseridosEmNoAlvo[j][i]+arc_Kde[i]

                else:
                    auxVar = self.__insertedArcs(arcos,elem[0])
                    auxVar = self.__probPaths(auxVar)
                    arcosInseridos = []
                    somaDosArcosInseridos = []
                    for i in auxVar:
                        arc_Kde = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde == None: return None
                        i.append(elem[1])
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        for i in range(len(arc_Kde)):
                            arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        arcosInseridos.append(arc_Kde)

                    auxVar = self.__insertedArcs(arcos,elem[1])
                    auxVar = self.__probPaths(auxVar)
                    for i in auxVar:
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        auxVar2 = [x for x in i if x != elem[1]]
                        arc_Kde3 = []
                        if len(auxVar2) == 1:
                            arc_Kde3 = probs.get(auxVar2[0])
                        else:
                            arc_Kde3 = self.__LOO_Kde(self.data, auxVar2, num_h)
                            if arc_Kde3 == None: return None
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
                    auxDictPar0 = probs.copy()
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
                    auxDictPar1 = probs.copy()
                    auxDictPar1.update({indicePar1[1]:resultadoPar1})
                    mlePar1 = []
                    for i in range(len(auxDictPar1.get(list(auxDictPar1.keys())[0]))):
                        mlePar1.append(1)
                        for j in range(len(auxDictPar1.keys())):
                            mlePar1[i] = mlePar1[i]*auxDictPar1.get(j)[i]
                        mlePar1[i] = math.log(mlePar1[i])
                    mlePar1 = sum(mlePar1)

                if mlePar0 != None and mlePar1 != None:
                    if mlePar0 > mlePar1:
                        if mlePar0 > last_MLE:
                            last_MLE = mlePar0
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, indicePar0)
                            probs = auxDictPar0.copy()
                            arcos.append(indicePar0) 
                    else:
                        if mlePar1 > last_MLE:
                            last_MLE = mlePar1
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, indicePar1)
                            probs = auxDictPar1.copy()
                            arcos.append(indicePar1)
        self.graphProbabilities = probs
        self.adjacencyMatrix = adjacency_matrix

    @property
    def graphProbabilities(self):
        """Returns all the probabilities of the graph.

        Returns:
            dictionary: The probabilities of the graph.
        """
        return self._graphProbabilities

    @property
    def adjacencyMatrix(self):
        """Returns the adjacency Matrix of the graph.

        Returns:
            list: the adjacency Matrix of the graph
        """
        return self._adjacencyMatrix

    def save(self, file_path):
        """Saves the network model into a file

        Args:
            file_path (string): The path of the file
        """
        pkl.dump(self,open(file_path,'wb'),-1)

    def load(file_path):
        """Loads the network model from a file

        Args:
            file_path (string): The path of the file

        Returns:
            BayesianNetwork: An object of the BayesianNetwork class
        """
        return pkl.load(open(file_path, 'rb'))