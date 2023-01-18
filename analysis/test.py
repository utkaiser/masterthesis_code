import scipy.io as sio
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def a():
    new_num_arr = np.load('../results/parareal/check_stability/diagonal.npy')
    plt.imshow(new_num_arr[4,0,0])
    plt.show()


if __name__ == '__main__':
    a()