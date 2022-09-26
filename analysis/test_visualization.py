import matplotlib.pyplot as plt
import numpy as np

def four_layers(x, y):
    res = x + np.pi / 3.1 * y
    if res < -1:
        return .2
    elif res < 0:
        return .6
    elif res < 1:
        return .8
    else:
        return 1

if __name__ == '__main__':
    dim = 128
    v_x = np.linspace(-1, 1, num=dim)
    v_y = np.linspace(-1, 1, num=dim)
    z = np.array([four_layers(i, j) for j in v_y for i in v_x])
    Z = np.array(z).reshape(dim, dim)
    plt.imshow(Z)
    plt.colorbar()
    plt.show()