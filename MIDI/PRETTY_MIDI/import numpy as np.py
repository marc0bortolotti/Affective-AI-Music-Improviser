import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

l = [a, b]

l+=[np.roll(a, 1)]

print(l)