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

print("MLE Independente: "+str(round(X1_mle*X2_mle, 2)))

X12 = np.column_stack((np.array(X1),np.array(X2))).tolist()
X12_h = obj.multivariateNewtonRaphson(X12, 0.9)

X12_mle = obj.maximumLikelihoodEstimation(X12, X12_h)/X2_mle
print("MLE 'A | B': "+str(round(X12_mle, 2)))

X21_mle = obj.maximumLikelihoodEstimation(X12, X12_h)/X1_mle
print("MLE 'B | A': "+str(round(X21_mle, 2)))