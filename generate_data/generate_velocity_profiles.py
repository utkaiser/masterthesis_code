import torch
import numpy as np
from scipy.io import loadmat
from skimage.filters import gaussian


def get_velocities(
        n_it,
        res_padded,
        velocity_profiles,
        optimization,
        boundary_condition
):

    if velocity_profiles == "bp_marmousi":
        input_path = f"../data/velocity_profiles/crops_bp_m_{n_it*2}_{res_padded}.npz"
        return np.load(input_path)['wavespeedlist']

    else:  # velocity_profiles == "mixed"
        input_path = f"../data/velocity_profiles/crops_bp_m_200_{res_padded}.npz"
        velocity_accumulated = np.load(input_path)['wavespeedlist']
        velocity_accumulated = np.concatenate((velocity_accumulated[:n_it],
                                               velocity_accumulated[100:100+n_it]),
                                              axis = 0)
        for vel_name in ["diagonal", "three_layers", "crack_profile", "high_frequency"]:
            crop = get_velocity_crop(res_padded, n_it, vel_name, boundary_condition, optimization)
            velocity_accumulated = np.concatenate((velocity_accumulated,
                                                   crop),
                                                  axis = 0)
        return velocity_accumulated

def get_velocity_crop(resolution, n_crops, velocity_profile, boundary_conditon, optimization):

    if velocity_profile == "diagonal":
        img = diagonal_ray(n_crops,resolution, boundary_conditon, optimization)

    elif velocity_profile == "marmousi":
        marmousi_datamat = loadmat('../data/velocity_profiles/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat['marm1larg'], 4)  # to make smoother
        img = np.expand_dims(marmousi_img[200:200+resolution,200:200+resolution], axis=0)

    elif velocity_profile == "marmousi2":
        marmousi_datamat = loadmat('../data/velocity_profiles/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat['marm1larg'], 4)  # to make smoother
        img = np.expand_dims(marmousi_img[300:300+resolution,300:300+resolution], axis=0)

    elif velocity_profile == "bp":
        databp = loadmat('../data/velocity_profiles/bp2004.mat')  # velocity models BP dataset
        img = gaussian(databp['V'], 4) / 1000  # to make smoother (and different order of magnitude)
        img = np.expand_dims(img[1100:1100 + resolution, 1100:1100 + resolution], axis=0)

    elif velocity_profile == "three_layers":
        img = three_layers(n_crops, resolution, boundary_conditon, optimization)

    elif velocity_profile == "crack_profile":
        img = crack_profile(n_crops, resolution, boundary_conditon, optimization)

    elif velocity_profile == "high_frequency":
        img = high_frequency(n_crops, resolution, boundary_conditon, optimization)

    else:
        raise NotImplementedError("Velocity model not implemented.")

    return img


def diagonal_ray(n_it, res, boundary_condition, optimization):
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if boundary_condition != "absorbing":
        factor = 2 if optimization == "parareal" else 1
        vel_profile = torch.from_numpy(3. + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.) > 0.3 / factor))
    else:
        if optimization == "none":
            vel_profile = torch.from_numpy(3. + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.) > 0.3/2))
        else:
            vel_profile = torch.from_numpy(3. + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.) > 0.3/3))

    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def three_layers(n_it, res, boundary_condition, optimization):

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if boundary_condition != "absorbing":
        factor = 2 if optimization == "parareal" else 1
        vel_profile = torch.from_numpy(2.6 + 0.0 * yy - .7 * (yy + xx - 0. > -.4/factor) - .7 * (yy + xx - 0. > .6/factor))
    else:
        if optimization == "none":
            vel_profile = torch.from_numpy(2.6 + 0.0 * yy - .7 * (yy + xx - 0. > -.4/2) - .7 * (yy + xx - 0. > .6/2))
        else:
            vel_profile = torch.from_numpy(
                2.6 + 0.0 * yy - .7 * (yy + xx - 0. > -.4 / 3) - .7 * (yy + xx - 0. > .6 / 3))
    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def crack_profile(n_it, res, boundary_condition, optimization):
    marmousi_datamat = loadmat('../data/velocity_profiles/marm1nonsmooth.mat')  # velocity models Marmousi dataset
    img = gaussian(marmousi_datamat['marm1larg'], 4)

    k1, k2, k3, k4 = .25, .5, .75, 1
    if boundary_condition != "absorbing":
        if res == 128:
            vel_profile = img[1100:1100 + res, 1100: 1100 + res]
            vel_profile[50:70,97:123] = k1
            vel_profile[10:28, 22:31] = k2
            vel_profile[60:118, 10:28] = k3
            vel_profile[100:118, 60:80] = k4

        else:  # res == 256:
            offset = 128//2 if optimization == "parareal" else 0
            vel_profile = img[900:900 + res, 900: 900 + res]
            vel_profile[50 + offset:70 + offset, 97 + offset:123 + offset] = k1
            vel_profile[10 + offset:28 + offset, 22 + offset:31 + offset] = k2
            vel_profile[60 + offset:118 + offset, 10 + offset:28 + offset] = k3
            vel_profile[100 + offset:118 + offset, 60 + offset:80 + offset] = k4

    elif boundary_condition == "absorbing":
        if optimization == "none":
            if res == 128 * 2:
                offset = 128//2
                vel_profile = img[1100:1100 + res, 1100: 1100 + res]
                vel_profile[50+offset:70+offset, 97+offset:123+offset] = k1
                vel_profile[10+offset:28+offset, 22+offset:31+offset] = k2
                vel_profile[60+offset:118+offset, 10+offset:28+offset] = k3
                vel_profile[100+offset:118+offset, 60+offset:80+offset] = k4

            else:  # res == 256 * 2:
                offset = 256//2
                vel_profile = img[900:900 + res, 900: 900 + res]
                vel_profile[100+offset:137+offset, 200+offset:245+offset] = k1
                vel_profile[18+offset:60+offset, 37+offset:60+offset] = k2
                vel_profile[120+offset:240+offset, 20+offset:60+offset] = k3
                vel_profile[195+offset:230+offset, 120+offset:160+offset] = k4
        else:
            if res == 128 * 3:
                offset = 128
                vel_profile = img[1100:1100 + res, 1100: 1100 + res]
                vel_profile[50 + offset:70 + offset, 97 + offset:123 + offset] = k1
                vel_profile[10 + offset:28 + offset, 22 + offset:31 + offset] = k2
                vel_profile[60 + offset:118 + offset, 10 + offset:28 + offset] = k3
                vel_profile[100 + offset:118 + offset, 60 + offset:80 + offset] = k4
            else:
                raise NotImplementedError("Resolution 256 x 256 for optimization and absorbing bc not implemented.")

    else: vel_profile = img

    return torch.from_numpy(vel_profile).unsqueeze(dim=0).repeat(n_it,1,1).numpy()


def high_frequency(n_it, res, boundary_condition, optimization):

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if boundary_condition == "absorbing":
        if optimization == "none":
            factor = 2
        else:
            factor = 3
    else:
        factor = 2 if optimization == "parareal" else 1

    vel_profile = torch.from_numpy(1. + 0.0 * yy)
    k = 0.03 / factor

    for i in range(res):
        if i < res//2:
            vel_profile[i:,i:] += k
        else:
            vel_profile[i:, i:] -= k

    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()