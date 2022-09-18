import numpy as np
import matplotlib.pyplot as plt

def waveguide(x,_):
    return .7 - .3 * np.cos(np.pi * x)

def inclusion(x,y):
    #0.7 + 0.05y + 0.1χ0.2<x<0.6,0.4<y<0.6
    res = .7 + .05*y
    tmp = .1 if .2 < x < .6 and .4 < y < .6 else 0
    return res + tmp

def fig9(x,y):
    band = .35
    if -band < y-x < band: return 1
    return 0.1

if __name__ == '__main__':
    dim = 128
    v_x = np.linspace(-1, 1, num=dim)
    v_y = np.linspace(-1, 1, num=dim)
    z = np.array([fig9(i, j) for j in v_y for i in v_x])
    Z = np.array(z).reshape(dim, dim)
    plt.contourf(v_y, v_x, Z, 100)
    plt.show()
