#Universidade Federal da Paraíba - Centro de Informática
# Author: Kelvin Brenand <brenand.kelvin@gmail.com>

import math
import copy
import random
import pickle
import json

MIN_VALUE = 0.001

class BayesianNetwork:
    '''
    This class implements the 1D and nD Newton-Raphson's bandwidth estimator method, its helping methods, and the 1D 
    and nD Kernel Density Estimation method.
    '''
    
    def __init__(self, data):
        self.data = data
        self.graphProbabilities = {}
        self.adjacencyMatrix = []

    @property
    def graphProbabilities(self):
        """Returns all the probabilities of the graph.

        Returns:
            dictionary: The probabilities of the graph.
        """
        return self._graphProbabilities

    @graphProbabilities.setter
    def graphProbabilities(self, value):
        """Sets a value to graphProbabilities.

        Args:
            value (dictionary): A dictionary to be assigned to graphProbabilities.
        """
        self._graphProbabilities = value

    @property
    def adjacencyMatrix(self):
        """Returns the adjacency Matrix of the graph.

        Returns:
            list: the adjacency Matrix of the graph
        """
        return self._adjacencyMatrix

    @adjacencyMatrix.setter
    def adjacencyMatrix(self, value):
        """Sets a value to adjacencyMatrix.

        Args:
            value (list): A list to be assigned to adjacencyMatrix.
        """
        self._adjacencyMatrix = value
        
    def __gaussian(self, x):
        """Compute the gaussian kernel of a given value.

        Args:
            x (float): Value to be computed.

        Returns:
            float: Gaussian kernel of the input value.
        """
        return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
    
    def __kernelDensityEstimation(self, x, data, h):
        """Using the gaussian kernel, it computes the Kernel Density Estimation (KDE) of the given data points and bandwidth parameter.

        Args:
            x (float): Point where the kde will be estimated.
            data (list): Data points to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            float: KDE of the input data and bandwidth.
        """
        if h <= 0: h = MIN_VALUE
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
        matrix = [[0. for _ in range(n)] for _ in range(n)]
        for i in range(0,n):
            matrix[i][i] = 1.
        return matrix

    def __ndim(self, x):
        """Number of array dimensions.

        Args:
            x (list): value that should have its number of dimensions found.

        Returns:
            int: Number of dimensions.
        """
        if isinstance(x[0], (float, int)): return 1
        return len(x[0])

    def __listSubtraction(self, x, y):
        """Performs the difference between two given lists.

        Args:
            x (list): First list.
            y (list): Second list.

        Returns:
            list: The difference between the two lists.
        """
        subtractionResult = []
        for i in range(len(x)):
            subtractionResult.append(x[i] - y[i])
        return subtractionResult

    def __listDivision(self, x, y):
        """Performs the division operation between the elements of two given lists.

        Args:
            x (list): First list.
            y (list): Second list.

        Returns:
            list: The result of the division.
        """
        divisionResult = []
        for i in range(len(x)):
            if y == 0: divisionResult.append(x[i]/MIN_VALUE)
            else: divisionResult.append(x[i]/y)
        return divisionResult

    def __dot(self, x,y):
        """Dot product of two values.

        Args:
            x (list): First list.
            y (list): Second list.

        Returns:
            list or float: If x and y are both NxN, it returns the square matrix multiplication.
                           If x is 1D and y is 2D, it returns the resulting list of the sum product.
                           If x and y are both 1D, it returns the resulting float of the inner product. 
        """
        if isinstance(x[0], list) and isinstance(y[0], list) and len(x[0]) == len(y[0]):
            dotResult = [[0 for _ in range(len(x[0]))] for _ in range(len(x[0]))]
            for i in range(len(x)):
                for j in range(len(y[0])):
                    for k in range(len(y)):
                        dotResult[i][j] += x[i][k] * y[k][j]
        elif not isinstance(x[0], list) and isinstance(y[0],list):
            dotResult = [0.] * len(x)
            for i in range(len(x)):
                for j in range(len(y)):
                    dotResult[i] += x[j] * y[j][i]
        else:
            dotResult = 0.
            for i in range(len(x)):
                dotResult += x[i]*y[i]
        return dotResult

    def __multivariateGaussian(self, x):
        """Compute the multivariate gaussian kernel of a given value. 

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
        """Using the multivariate gaussian kernel, it computes the Multivariate Kernel Density Estimation (MKDE) of 
        the given data points and bandwidth parameter.

        Args:
            x (list): N dimensional point where the mKDE will be estimated.
            data (list): Data points to compute the KDE from.
            h (float): Bandwidth parameter.

        Returns:
            float: KDE of the input data and bandwidth.
        """

        sum = 0
        for i in range(len(data)):
            sum += self.__multivariateGaussian(self.__listDivision(self.__listSubtraction(x, data[i]), h))
        if (len(data)*h**self.__ndim(x)) == 0: return (sum/MIN_VALUE)
        else: return (sum/(len(data)*h**self.__ndim(x)))

    def __intervalH(self, data):
        """Compute the right endpoint of the interval of values h can assume. The other endpoint is 0.1.

        Args:
            data (List): Data points to compute the interval from.

        Returns:
            float: The right endpoint.
        """
        minPointsDist = []
        for i in data:
            pointDist = []
            for j in data:
                if i == j: continue
                if isinstance(i, (float, int)): pointDist.append(math.dist([i],[j]))
                else: pointDist.append(math.dist(i,j))
            if pointDist == []: minPointsDist.append(MIN_VALUE)
            else: minPointsDist.append(min(pointDist))
        return max(minPointsDist)

    def __LOO_Kde(self, data, index, num_h):
        """Performs the bandwidth estimation and the Leave-One-Out KDE for either the 1D KDE or the Multivariate KDE.

        Args:
            data (List): Data points to compute the KDE from.
            index (tuple, int): The index of columns to use in the data partition method.
            num_h (int): The amount of h to be used to compute the best KDE. 

        Returns:
            list: KDE of myData and bandwidth h.
        """
        dataSegment = self.__dataPartition(data, index)
        rightEndpoint = self.__intervalH(dataSegment)
        hs = {round(random.uniform(0.1, rightEndpoint),1) for _ in range(num_h)}
        logsAndKdes = {}
        for h in hs:
            kde = []
            dataSegmentCopy = dataSegment.copy()
            for i in range(len(dataSegment)):
                element = dataSegmentCopy[i]
                dataSegmentCopy.pop(i)
                if isinstance(element, (float, int)): kde.append(self.__kernelDensityEstimation(element, dataSegmentCopy, h))
                else: kde.append(self.__multivariateKernelDensityEstimation(element, dataSegmentCopy, h))
                dataSegmentCopy.clear()
                dataSegmentCopy = dataSegment.copy()
            log = []
            for i in range(len(kde)):
                if kde[i] <= 0: log.append(math.log(MIN_VALUE))
                else: log.append(math.log(kde[i]))
            logsAndKdes.update({sum(log):kde})
        return logsAndKdes.get(max(logsAndKdes))

    def __pairs(self, n):
        """Returns all possible pairs from 0 up to n. E.g.: For n = 2, it will return [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)].

        Args:
            n (int): The upper limit from where the method will generate the pairs.

        Returns:
            list of tuples: The list of all the pairs.
        """
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i,j))
                pairs.append((j,i))
        return pairs
    
    def __dataPartition(self,data,index):
        """Returns n specific columns from a N dimensional list.

        Args:
            data (list): The complete initial list.
            index (tuple,int): The n specific columns.

        Returns:
            list: The specific columns of the N dimensional data.
        """
        if isinstance(index, int): return [row[index] for row in data]
        else:
            partitionResult = []
            for i in range(len(data)):
                dataSlice = []
                for j in index:
                    dataSlice.append(data[i][j])
                partitionResult.append(dataSlice)
            return partitionResult

    def __mtxModifier(self, matrix, index):
        """Modifies the adjacency matrix that will be returned by the MLE method.

        Args:
            matrix (list): Initial adjacency matrix.
            index (list): The index that will be modified in the adjacency matrix.

        Returns:
            list: The modified adjacency matrix.
        """
        matrix[index[0]][index[1]] = 1
        return matrix

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
        for _ in range(1,len(matrix[0])):
            An = self.__dot(An, matrix)
            if self.__trace(An) != 0: return True
        return False

    def __insertedArcs(self,arcs, elem):
        """Returns a dictionary with all nodes associated to elem.

        Args:
            arcs (list): List with all the nodes and all the arcs.
            elem (int): Target node

        Returns:
            dictionary: All nodes associated to elem.
        """
        elemRelatedArcs = {}
        aux = []
        for i in arcs:
            if isinstance(i, tuple):
                if i[1] == elem: aux.append(i[0])
        elemRelatedArcs.update({elem:aux})
        for j in aux:
            elemRelatedArcs.update(self.__insertedArcs(arcs,j))
        return elemRelatedArcs

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
            if arcs.get(i) == []: myDict.pop(i)
        for i in myDict.keys():
            for j in myDict.get(i):
                myList1.append([i,j])
        
        if len(myList1) == 1: return myList1
        myList2 = []
        for i in myList1:
            flag = True
            for j in myList1:
                if i[-1] == j[0]:
                    flag = False
                    aux = i.copy()
                    aux.append(j[1])
                    myList2.append(aux)
            if flag: myList2.append(i)

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

    def getGraph(self, num_h=10):
        """Compute the Maximum-Likelihood Estimation (MLE) of data and returns the adjacency matrix.

        Args:
            data (list): Data points to compute the MLE from.
            num_h (int, optional): The amount of h to be used to compute the best KDE. Defaults to 10.

        Raises:
            TypeError: data must be list.

        Returns:
            list: The adjacency matrix associated with data.
            None: If a RuntimeError happened.
        """

        if not (isinstance(self.data, list) and isinstance(self.data[0], list)):
            raise TypeError("data must be list of lists")

        probs = {}
        adjacency_matrix = [[0 for _ in range(len(self.data[0]))] for _ in range(len(self.data[0]))]
        for i in range (len(self.data[0])):
            nodeKde = self.__LOO_Kde(self.data, i, num_h)
            if nodeKde == None: return None
            probs.update({(i):nodeKde})
        auxList = []
        
        probsMultiplication = 1.0
        for i in range(len(probs.get((0)))):
            for j in range(len(probs)):
                probsMultiplication = probsMultiplication*probs.get((j))[i]
                if probsMultiplication == 0: probsMultiplication = MIN_VALUE
            auxList.append(math.log(probsMultiplication))
            probsMultiplication = 1.0
        last_MLE = sum(auxList)

        arcs = list(probs.keys())
        indices = self.__pairs(len(self.data[0]))
        indicesCopy = indices.copy() 
        for elem in indices:
            if list(self.__insertedArcs(arcs,elem[1]).values()) == [[]]: 
                if list(self.__insertedArcs(arcs,elem[0]).values()) == [[]]:
                    arc_Kde = self.__LOO_Kde(self.data, elem, num_h)
                    if arc_Kde == None: return None
                    for i in range(len(arc_Kde)):
                        if probs.get(elem[0])[i] == 0: arc_Kde[i] = arc_Kde[i]/MIN_VALUE
                        else: arc_Kde[i]/probs.get(elem[0])[i]
                    sumInsertedArcs = arc_Kde.copy()
                
                else:
                    nodeProbPaths = self.__probPaths(self.__insertedArcs(arcs,elem[0]))
                    insertedArcs = []
                    sumInsertedArcs = []
                    for i in nodeProbPaths:
                        arc_Kde = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde == None: return None
                        i.append(elem[1])
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        for i in range(len(arc_Kde)):
                            if arc_Kde[i] == 0: arc_Kde[i] = arc_Kde2[i]/MIN_VALUE
                            else: arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        insertedArcs.append(arc_Kde)

                    for i in range(len(insertedArcs[0])):
                        sumInsertedArcs.append(0)
                        for j in range(len(insertedArcs)):
                            sumInsertedArcs[i] = sumInsertedArcs[i]+insertedArcs[j][i]
            else:
                if list(self.__insertedArcs(arcs,elem[0]).values()) == [[]]:
                    arc_Kde = self.__LOO_Kde(self.data, elem, num_h)
                    if arc_Kde == None: return None
                    for i in range(len(arc_Kde)):
                        if probs.get(elem[0])[i] == 0: arc_Kde[i] = arc_Kde[i]/MIN_VALUE
                        else: arc_Kde[i] = arc_Kde[i]/probs.get(elem[0])[i]

                    nodeProbPaths = self.__probPaths(self.__insertedArcs(arcs,elem[1]))
                    arcsAlreadyInTargetNode = []
                    sumInsertedArcs = []
                    for i in nodeProbPaths:
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        index = [x for x in i if x != elem[1]]
                        arc_Kde3 = []
                        if len(index) == 1: arc_Kde3 = probs.get(index[0])
                        else:
                            arc_Kde3 = self.__LOO_Kde(self.data, index, num_h)
                            if arc_Kde3 == None: return None
                        for i in range(len(arc_Kde)):
                            if arc_Kde3[i] == 0: arc_Kde2[i] = arc_Kde2[i]/MIN_VALUE
                            else: arc_Kde2[i] = arc_Kde2[i]/arc_Kde3[i]
                        arcsAlreadyInTargetNode.append(arc_Kde2)
                    
                    for i in range(len(arcsAlreadyInTargetNode[0])):
                        sumInsertedArcs.append(0)
                        for j in range(len(arcsAlreadyInTargetNode)):
                            sumInsertedArcs[i] = sumInsertedArcs[i]+arcsAlreadyInTargetNode[j][i]+arc_Kde[i]

                else:
                    nodeProbPaths = self.__probPaths(self.__insertedArcs(arcs,elem[0]))
                    insertedArcs = []
                    sumInsertedArcs = []
                    for i in nodeProbPaths:
                        arc_Kde = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde == None: return None
                        i.append(elem[1])
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        for i in range(len(arc_Kde)):
                            if arc_Kde[i] == 0: arc_Kde[i] = arc_Kde2[i]/MIN_VALUE
                            else: arc_Kde[i] = arc_Kde2[i]/arc_Kde[i]
                        insertedArcs.append(arc_Kde)

                    nodeProbPaths = self.__probPaths(self.__insertedArcs(arcs,elem[1]))
                    for i in nodeProbPaths:
                        arc_Kde2 = self.__LOO_Kde(self.data, i, num_h)
                        if arc_Kde2 == None: return None
                        index = [x for x in i if x != elem[1]]
                        arc_Kde3 = []
                        if len(index) == 1:
                            arc_Kde3 = probs.get(index[0])
                        else:
                            arc_Kde3 = self.__LOO_Kde(self.data, index, num_h)
                            if arc_Kde3 == None: return None
                        for i in range(len(arc_Kde)):
                            if arc_Kde3[i] == 0: arc_Kde2[i] = arc_Kde2[i]/MIN_VALUE
                            else: arc_Kde2[i] = arc_Kde2[i]/arc_Kde3[i]
                        insertedArcs.append(arc_Kde2)
                    
                    for i in range(len(insertedArcs[0])):
                        sumInsertedArcs.append(0)
                        for j in range(len(insertedArcs)):
                            sumInsertedArcs[i] = sumInsertedArcs[i]+insertedArcs[j][i]
            
            try:
                indicesCopy.index((elem[1],elem[0]))
                firstPairIndex = elem
                firstPairResult = sumInsertedArcs.copy()
                indicesCopy.remove(elem)
            except:
                secondPairIndex = elem
                secondPairResult = sumInsertedArcs.copy()
                mtxCopy = copy.deepcopy(adjacency_matrix)
                mtxCopy = self.__mtxModifier(mtxCopy,firstPairIndex)
                mlePar0 = None
                if not self.__cycle(mtxCopy):
                    auxDictPar0 = probs.copy()
                    auxDictPar0.update({firstPairIndex[1]:firstPairResult})
                    mlePar0 = []
                    for i in range(len(auxDictPar0.get(list(auxDictPar0.keys())[0]))):
                        mlePar0.append(1)
                        for j in range(len(auxDictPar0.keys())):
                            mlePar0[i] = mlePar0[i]*auxDictPar0.get(j)[i]
                        if mlePar0[i] <= 0: mlePar0[i] = math.log(MIN_VALUE)
                        else: mlePar0[i] = math.log(mlePar0[i])
                    mlePar0 = sum(mlePar0)
                
                mtxCopy = copy.deepcopy(adjacency_matrix)
                mtxCopy = self.__mtxModifier(mtxCopy,secondPairIndex)
                mlePar1 = None
                if not self.__cycle(mtxCopy):
                    auxDictPar1 = probs.copy()
                    auxDictPar1.update({secondPairIndex[1]:secondPairResult})
                    mlePar1 = []
                    for i in range(len(auxDictPar1.get(list(auxDictPar1.keys())[0]))):
                        mlePar1.append(1)
                        for j in range(len(auxDictPar1.keys())):
                            mlePar1[i] = mlePar1[i]*auxDictPar1.get(j)[i]
                        if mlePar1[i] <= 0: mlePar1[i] = math.log(MIN_VALUE)
                        else: mlePar1[i] = math.log(mlePar1[i])
                    mlePar1 = sum(mlePar1)

                if mlePar0 != None and mlePar1 != None:
                    if mlePar0 > mlePar1:
                        if mlePar0 > last_MLE:
                            last_MLE = mlePar0
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, firstPairIndex)
                            probs = auxDictPar0.copy()
                            arcs.append(firstPairIndex) 
                    else:
                        if mlePar1 > last_MLE:
                            last_MLE = mlePar1
                            adjacency_matrix = self.__mtxModifier(adjacency_matrix, secondPairIndex)
                            probs = auxDictPar1.copy()
                            arcs.append(secondPairIndex)
        self.graphProbabilities = probs
        self.adjacencyMatrix = adjacency_matrix

    def __predictPoint(self, point, num_h):
        """Compute the probability that the point belongs to a class.

        Args:
            point (list): N dimensional point where the mKDE will be estimated.
            num_h (int): The amount of h to be used to compute the best KDE.

        Returns:
            float: The probability of belonging.
        """
        rightEndpoint = self.__intervalH(self.data)
        hs = {round(random.uniform(0.1, rightEndpoint),1) for _ in range(num_h)}
        kde = []
        for h in hs:
            kde.append(self.__multivariateKernelDensityEstimation(point, self.data, h))
        return round(max(kde),3)

    @staticmethod
    def predict(x_test, classLabels, networks, num_h=10):
        """Predict the class for the provided data.

        Args:
            x_test (list): Test samples.
            classLabels (list): The classes labels.
            networks (list of BayesianNetwork): The classes.
            num_h (int, optional): The amount of h to be used to compute the best KDE. Defaults to 10.

        Returns:
            list: Class labels for each data sample
        """
        y_pred = []
        for i in x_test:
            classPrediction = []
            for j in networks:
                classPrediction.append(j.__predictPoint(i, num_h))
            y_pred.append(classLabels[classPrediction.index(max(classPrediction))])
        return y_pred

    @staticmethod
    def kfoldcv(data, labels, k = 10, num_h=10, accuracy = False, confMtx = False):
        """Perform the Kfold Cross Validation to obtain the best group of networks based on its accuracy value.

        Args:
            data (list): The data points.
            labels (list): The classes labels.
            k (int, optional): Number of folds. Defaults to 10.
            num_h (int, optional): The amount of h to be used to compute the best KDE. Defaults to 10.
            accuracy (bool, optional): Return the accuracy if true. Defaults to False.
            confMtx (bool, optional): Return the confusion matrix if true. Defaults to False.

        Raises:
            ValueError: Number of folds k must be greater than 1 and smaller than the data size.

        Returns:
            list of BayesianNetwork: The best group of networks.
        """

        size = len(data)
        if k <=1 or k > size: raise ValueError('Invalid k value')
        subset_size = round(size / k)
        temp = list(zip(data, labels))
        random.shuffle(temp)
        data, labels = zip(*temp)
        data = list(data)
        labels = list(labels)
        count = 0
        netsAndYpredtestAndAcc = []
        for x in range(0, size, subset_size):
            count += 1
            if count > k: break
            X_train = data[:x]+data[x+subset_size:]
            y_train = labels[:x]+labels[x+subset_size:]
            X_test = data[x:x+subset_size]
            y_test = labels[x:x+subset_size]
            
            classes = list(set(y_train))
            networks = []
            for j in classes:
                networks.append(BayesianNetwork([X_train[i] for i in range(len(y_train)) if y_train[i] == j]))
            y_pred = BayesianNetwork.predict(X_test, classes, networks, num_h)
            netsAndYpredtestAndAcc.append((networks, y_test, y_pred, BayesianNetwork.accuracy(y_test, y_pred)))
        idxAndValueBestAcc = [0,0]
        for i in range(len(netsAndYpredtestAndAcc)):
            if netsAndYpredtestAndAcc[i][-1] > idxAndValueBestAcc[-1]:
                idxAndValueBestAcc[0] = i
                idxAndValueBestAcc[-1] = netsAndYpredtestAndAcc[i][-1]

        if accuracy and confMtx:
            return netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][0], netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][-1], BayesianNetwork.confusionMatrix(netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][1],netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][2])
        if accuracy and not confMtx:
            return netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][0], netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][-1]
        if confMtx and not accuracy:
            return netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][0], BayesianNetwork.confusionMatrix(netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][1],netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][2])
        return netsAndYpredtestAndAcc[idxAndValueBestAcc[0]][0]

    @staticmethod
    def accuracy(actual, predicted):
        """Accuracy classification score.

        Args:
            actual (list): Correct labels.
            predicted (list): Predicted labels.

        Returns:
            float: The percentage of correctly classified samples.
        """
        totalElements = len(actual)
        correctPredictions = sum([1 for i in range(totalElements) if actual[i] == predicted[i]])
        return round(correctPredictions/totalElements,2)

    @staticmethod
    def confusionMatrix(actual, predicted):
        """Calculate the confusion matrix to evaluate classification accuracy.

        Args:
            actual (list): Correct labels.
            predicted (list): Predicted labels.

        Returns:
            list: The confusion matrix.
        """
        unique = sorted(set(actual))
        matrix = [[0 for _ in unique] for _ in unique]
        imap   = {key: i for i, key in enumerate(unique)}
        for a, p in zip(actual, predicted):
            matrix[imap[a]][imap[p]] += 1
        return matrix

    def save(self, file_path):
        """Saves the network model into a pickle file

        Args:
            file_path (string): The path of the file
        """
        pickle.dump(self,open(file_path,'wb'),-1)

    def load(file_path):
        """Loads the network model from a pickle file

        Args:
            file_path (string): The path of the file

        Returns:
            BayesianNetwork: An object of the BayesianNetwork class
        """
        return pickle.load(open(file_path, 'rb'))

    def saveJson(self, file_path):
        """Saves the network model into a json file

        Args:
            file_path (string): The path of the file
        """
        with open(file_path, 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4)