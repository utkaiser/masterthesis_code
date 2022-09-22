import numpy as np
import matplotlib.pyplot as plt

def waveguide(x,_):
    return .7 - .3 * np.cos(np.pi * x)

def inclusion(x,y):
    res = .7 + .05*y
    tmp =  0.1*(np.abs(x-0.4)<0.2)*(np.abs(y-0.5)<0.1)
    return res + tmp

def fig9(x,y):
    band = .35
    if np.logical_and(-band < y-x, y-x < band):
        return 1
    return 0.5

def other(x,y):
    return np.cos(8*np.pi*y) * np.exp(-25*(x**2)-250*(y**2))

if __name__ == '__main__':
    dim = 128
    x = np.linspace(-1, 1, num=dim)
    y = np.linspace(-1, 1, num=dim)
    #X, Y = np.meshgrid(x, y)
    Z = np.array([other(i,j) for j in y for i in x])
    Z = Z.reshape(dim,dim)
    plt.contourf(y, x, Z, 100)
    plt.colorbar()
    plt.show()
