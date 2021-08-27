from functions import functions
import numpy as np

DATA_SIZE = 2000
np.random.seed(seed=123)

data = np.random.randn(DATA_SIZE).tolist()+(np.random.randn(DATA_SIZE)*np.sqrt(1)+3).tolist()
obj = functions()

best_h = obj.nrmloo(data, 1.0)
print(best_h)