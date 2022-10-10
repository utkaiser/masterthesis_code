import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = np.ones((10,10))
    a = np.pad(a, ((1, 1), (1, 1)), mode='constant', constant_values=((10, 10), (10, 10)))
    a = np.roll(a, 1, axis=1) - 2 * a + np.roll(a, -1, axis=1) + (np.roll(a, 1, axis=0) - 2 * a + np.roll(a, -1, axis=0))
    print(a[1:-1,1:-1])








