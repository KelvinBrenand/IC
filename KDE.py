import math

def gaussian(x):
    return ((1/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2)))
def kde(x, array, h):
    N = len(array)
    sum = 0
    for i in range(N):
        sum += gaussian((x - array[i])/h)
    return sum/(N*h)