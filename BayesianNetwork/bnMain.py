import math
from sklearn import datasets
from bn import newtonRapson
import numpy as np

data = datasets.load_iris()
data1 = data.data[:,:2].tolist()
print(np.array(data1).shape)
obj = newtonRapson()
X = obj.MLE(data1)
print(np.array(X))
#print("MLE Independente: "+str(round(X[0], 2)))

# X1 = data.data[:, :1].tolist()
# X1 = [item for sublist in X1 for item in sublist]

# X2 = data.data[:, 1:2].tolist()
# X2 = [item for sublist in X2 for item in sublist]

# obj = newtonRapson()
# X1_h = obj.newtonRaphson(X1, 0.7)

# X2_h = obj.newtonRaphson(X2, 0.5)

# X1_kde = obj.LOO_Kde(X1, X1_h)

# X2_kde = obj.LOO_Kde(X2, X2_h)

# auxx=[]
# for i in range(len(X1_kde)):
#     auxx.append(math.log(X1_kde[i]*X2_kde[i]))
# X3 = sum(auxx)

# print("MLE Independente: "+str(round(X3, 2)))



#X12 = np.column_stack((np.array(X1),np.array(X2))).tolist()

# X12_h = obj.multivariateNewtonRaphson(X12, 0.9)

# X12_kde = obj.LOO_Kde(X12, X12_h)

# auxx=[]
# for i in range(len(X1_kde)):
#     auxx.append(math.log(X12_kde[i]/X2_kde[i]))
# XX12 = sum(auxx)

# print("MLE 'A | B': "+str(round(XX12, 2)))

# auxx=[]
# for i in range(len(X1_kde)):
#     auxx.append(math.log(X12_kde[i]/X1_kde[i]))
# XX21 = sum(auxx)


# print("MLE 'B | A': "+str(round(XX21, 2)))