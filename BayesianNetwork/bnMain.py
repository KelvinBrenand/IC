from sklearn import datasets
from nrbe import newtonRapson

iris = datasets.load_iris()

X1 = iris.data[:, :1].tolist()
X1 = [item for sublist in X1 for item in sublist]

X2 = iris.data[:, 1:2].tolist()
X2 = [item for sublist in X2 for item in sublist]

obj = newtonRapson()
X1_h = obj.newtonRaphson(X1, 0.7)

X2_h = obj.newtonRaphson(X2, 0.5)

X1_mle = obj.maximumLikelihoodEstimation(X1, X1_h)
print("Maximum Likelihood Estimation da primeira coluna: "+str(round(X1_mle, 2)))

X2_mle = obj.maximumLikelihoodEstimation(X2, X2_h)
print("Maximum Likelihood Estimation da segunda coluna: "+str(round(X2_mle, 2)))

print("Maximum Likelihood Estimation das colunas serem independentes: "+str(round(X1_mle*X2_mle, 2)))