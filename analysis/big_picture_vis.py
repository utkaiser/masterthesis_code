import numpy as np
import torch
import matplotlib.pyplot as plt
from generate_data.generate_velocity_crop import createCropsAndSave
from scipy.io import loadmat
from skimage.filters import gaussian
from generate_data.wave_util import crop_center

def visualize_big_picture(res_x=1500, res_y=700, n_inits=10, velocity_crop="marmousi", solver="vv_absorbing"):

    # params setting
    Tf = 2.0
    cT = 0.2
    dx = 2.0 / 128.0
    dt = dx / 20
    ny, nx = 64, 64
    m = 2
    rt = 4
    mt = round(Tf / cT)
    t = np.linspace(0, Tf, mt)
    x = np.arange(-1, 1, dx)
    y = np.arange(-1, 1, dx)
    xx, yy = np.meshgrid(x, y)

    #init condition
    np.random.seed = 21
    center = np.array([0., 0.])
    # center_1 = np.array([-0.8,-0.8])
    # u0_1 = np.exp(-250.0*(0.2*(xx-center_1[0])**2 + (yy-center_1[1])**2))*np.cos(8*np.pi*(yy-center_1[1]))
    # center_2 = np.array([.8,.8])
    # u0_2 = np.exp(-250.0*(0.2*(xx-center_2[0])**2 + (yy-center_2[1])**2))*np.cos(8*np.pi*(yy-center_2[1]))
    # u0 = u0_1 + u0_2
    u0 = np.exp(-250.0 * (0.2 * (xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * np.cos(8 * np.pi * (yy - center[1]))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])

    datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
    fullmarm = gaussian(datamat['marm1larg'], 4)  # to make smoother
    vel = crop_center(fullmarm, res_x, res_y)
    plt.imshow(vel)
    plt.show()

    #TODO: get initial condition
    #TODO: velocity verlet iteration
    #TODO: visualization




if __name__ == '__main__':
    visualize_big_picture()