from functions import functions
import numpy as np
import matplotlib.pyplot as plt

DATA_SIZE = 600
np.random.seed(seed=123)

data = np.random.randn(DATA_SIZE).tolist()
obj = functions()

best_h = obj.nrmloo(data, 1.0)
print(best_h)