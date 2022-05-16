import math
from sklearn import datasets
from bn import newtonRapson
import numpy as np

iris = datasets.load_iris()

X1 = iris.data[:, :1].tolist()
X1 = [item for sublist in X1 for item in sublist]

X2 = iris.data[:, 1:2].tolist()
X2 = [item for sublist in X2 for item in sublist]

obj = newtonRapson()
X1_h = obj.newtonRaphson(X1, 0.7)

X2_h = obj.newtonRaphson(X2, 0.5)

X1_mle = obj.maximumLikelihoodEstimation(X1, X1_h)

X2_mle = obj.maximumLikelihoodEstimation(X2, X2_h)

X1_mle = np.array(X1_mle)
X2_mle = np.array(X2_mle)
X3 = X1_mle*X2_mle
X3 = np.log(X3)
X3 = sum(X3)
print("MLE Independente: "+str(round(X3, 2)))

X12 = np.column_stack((np.array(X1),np.array(X2))).tolist()
X12_h = obj.multivariateNewtonRaphson(X12, 0.9)

X12_mle = obj.maximumLikelihoodEstimation(X12, X12_h)
X12_mle = np.array(X12_mle)
XX12 = X12_mle/X2_mle
XX12 = np.log(XX12)
XX12 = sum(XX12)

print("MLE 'A | B': "+str(round(XX12, 2)))

X21_mle = X12_mle
XX21 = X21_mle/X1_mle
XX21 = np.log(XX21)
XX21 = sum(XX21)


print("MLE 'B | A': "+str(round(XX21, 2)))