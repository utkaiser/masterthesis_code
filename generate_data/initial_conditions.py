import numpy as np
from generate_data.utils_wave import WaveEnergyComponentField_tensor
import random
from scipy.io import loadmat
from skimage.filters import gaussian


def init_cond_gaussian(vel, res = 128, boundary_condition="absorbing", mode = "generate_data", res_padded = 128 * 3):

    dx, width, center_x, center_y = get_init_cond_settings(res, boundary_condition)
    xx, yy = np.meshgrid(np.linspace(-1, 1, res_padded), np.linspace(-1, 1, res_padded))
    u0 = torch.from_numpy(np.exp(-width * ((xx - center_x) ** 2 + (yy - center_y) ** 2))).unsqueeze(dim=0)
    ut0 = torch.from_numpy(np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])).unsqueeze(dim=0)

    if mode == "generate_data":
        return u0, ut0
    elif mode == "parareal":
        wx, wy, wtc = WaveEnergyComponentField_tensor(u0, ut0, vel.unsqueeze(dim=0), dx=dx)
        return torch.stack([wx, wy, wtc], dim=1)
    else: raise NotImplementedError("Mode for initial condition not implemented")


def get_init_cond_settings(res, boundary_condition):

    factor_width = random.random() * 2 - 1
    factor_center_x = random.random() * 2 - 1
    factor_center_y = random.random() * 2 - 1

    if boundary_condition == "periodic":
        center_x, center_y = factor_center_x * .4, factor_center_y * .4
        if res == 128:
            dx = 2.0 / 128.0
            width = 1000 + factor_width*200
        elif res == 256:
            dx = 2.0 / 256.0
            width = 2000 + factor_width*300
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    elif boundary_condition == "absorbing":
        center_x, center_y = (factor_center_x * .4) / 3, (factor_center_y * .4) / 3
        if res == 128:
            dx = 2.0 / (128.0 * 3)
            width = 2300 + factor_width*300
        elif res == 256:
            dx = 2.0 / (256.0 * 3)
            width = 4200 + factor_width*500
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    else: raise NotImplementedError("Boundary condition for initial condition not implemented.")

    return dx, width, center_x, center_y


def init_cond_ricker(xx, yy, width, center):
    """
    Ricker pulse wavefield
    """

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    u0 = (1 - 2 * width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * u0
    u0 = u0 / np.max(np.abs(u0))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0

import torch

def diagonal_ray(n_it, res):

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    vel_profile = torch.from_numpy(3. + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.) > 0.3))
    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()

def three_layers(n_it, res):
    #  x+pi/3.1*y-l_k

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    vel_profile = torch.from_numpy(2.2 + 0.0 * yy - .7 * (yy + xx - 0. > -.4) - .7 * (yy + xx - 0. > .6))
    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def crack_profile(n_it, res):
    marmousi_datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
    vel_profile = gaussian(marmousi_datamat['marm1larg'], 4)[1100:1100 + res, 1100:1100 + res]
    k1, k2, k3, k4 = .25, .5, .75, 1

    if res == 128:
        vel_profile[50:70,97:123] = k1
        vel_profile[10:28, 22:31] = k2
        vel_profile[60:118, 10:28] = k3
        vel_profile[100:118, 60:80] = k3
    if res == 256:
        vel_profile[50:70, 97:123] = k1
        vel_profile[10:28, 22:31] = k2
        vel_profile[60:118, 10:28] = k3
        vel_profile[100:118, 60:80] = k3
    if res == 128 * 3:
        offset = 128 * 3 // 4
        vel_profile[50+offset:70+offset, 97+offset:123+offset] = k1
        vel_profile[10+offset:28+offset, 22+offset:31+offset] = k2
        vel_profile[60+offset:118+offset, 10+offset:28+offset] = k3
        vel_profile[100+offset:118+offset, 60+offset:80+offset] = k3
    if res == 256 * 3:
        pass

    return torch.from_numpy(vel_profile).unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def high_frequency(n_it, res):
    #  x+pi/3.1*y-l_k

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    vel_profile = torch.from_numpy(1. + 0.0 * yy)
    k = 0.05
    for i in range(res):
        if i < 100:
            vel_profile[i:,i:] += k
        else:
            vel_profile[i:, i:] -= k

    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def get_velocity_crop(resolution, n_crops, velocity_profile):

    if velocity_profile == "diagonal":
        img = diagonal_ray(n_crops,resolution)

    elif velocity_profile == "marmousi":
        marmousi_datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat['marm1larg'], 4)  # to make smoother
        img = marmousi_img[200:200+resolution,200:200+resolution].unsqueeze(dim=0)

    elif velocity_profile == "marmousi2":
        marmousi_datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat['marm1larg'], 4)  # to make smoother
        img = marmousi_img[300:300+resolution,300:300+resolution].unsqueeze(dim=0)

    elif velocity_profile == "bp":
        databp = loadmat('../data/bp2004.mat')  # velocity models BP dataset
        img = gaussian(databp['V'], 4) / 1000  # to make smoother (and different order of magnitude)
        img = img[1100:1100 + resolution, 1100:1100 + resolution].unsqueeze(dim=0)

    elif velocity_profile == "three_layers":
        img = three_layers(n_crops, resolution)

    elif velocity_profile == "crack_profile":
        img = crack_profile(n_crops, resolution)

    elif velocity_profile == "high_frequency":
        img = high_frequency(n_crops, resolution)

    else:
        raise NotImplementedError("Velocity model not implemented.")

    return img

import matplotlib.pyplot as plt

def get_velocities(n_it, input_path, res, boundary_condition, n_crops = 50):

    if boundary_condition == "absorbing": res_padded = res * 3
    else: res_padded = res

    velocities = np.concatenate([
        # get_velocity_crop(res, n_crops, "diagonal"),  # n_crops x res x res
        # get_velocity_crop(res, n_crops, "three_layers"),  # n_crops x res x res
        get_velocity_crop(res, n_crops, "crack_profile"),  # n_crops x res x res
        # get_velocity_crop(res, n_crops, "high_frequency"),  # n_crops x res x res
        # np.load(input_path)['wavespeedlist']  # 200 x 2000 x 2000
    ], axis=0)

    for i in range(10):
        v = velocities[i]
        plt.imshow(v)
        plt.colorbar()
        plt.show()

    if n_it == -1:
        n_it = velocities.shape[0]
    else:
        idx = np.random.permutation(velocities.shape[0])[:n_it]
        velocities = velocities[idx]

    return velocities.shape, n_it, res_padded


if __name__ == '__main__':
    print(get_velocities(10,"../data/crops_bp_m_200_2000.npz", 128 * 3, "absorbing"))






