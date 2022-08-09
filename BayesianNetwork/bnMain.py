from bn import BayesianNetwork
import numpy as np
import pandas as pd
import time

def formattedTime(seconds):
    hours = round(seconds/3600)
    minutes = round((seconds%3600)/60)
    seconds = round((seconds%3600)%60)
    print("Time spent: "+str(hours)+"h:"+str(minutes)+"m:"+str(seconds)+"s")

df = pd.read_csv('BayesianNetwork\Algerian_forest_fires.csv')
df.dropna()
X = df.iloc[:, :-1].values.tolist()
y = df.iloc[:, -1].values.tolist()

beginningTime = time.time()
networks, acc, confMtx = BayesianNetwork.kfoldcv(X, y, k=10, num_h=20, accuracy=True, confMtx=True)
endTime = time.time()
formattedTime(endTime-beginningTime)

print("Accuracy:",acc)
print(np.array(confMtx),"\n")

# for net in networks:
#     print(np.array(net.adjacencyMatrix),"\n")