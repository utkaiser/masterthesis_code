import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
from generate_data.utils_wave import WaveEnergyComponentField_tensor
import random
from scipy.io import loadmat
from skimage.filters import gaussian
import logging
import torch


def initial_condition_gaussian(vel, res, boundary_condition, res_padded, optimization, mode="generate_data"):
    dx, width, center_x, center_y = get_init_cond_settings(res, boundary_condition, optimization)
    xx, yy = np.meshgrid(np.linspace(-1, 1, res_padded), np.linspace(-1, 1, res_padded))
    u0 = np.exp(-width * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])

    if mode == "generate_data":
        return u0, ut0
    else:  # "parareal" or "other"
        u0, ut0 = torch.from_numpy(u0).unsqueeze(dim=0), torch.from_numpy(ut0).unsqueeze(dim=0)
        wx, wy, wtc = WaveEnergyComponentField_tensor(u0, ut0, vel.unsqueeze(dim=0), dx=dx)
        return torch.stack([wx, wy, wtc], dim=1)

def get_init_cond_settings(res, boundary_condition, optimization):

    factor_width = random.random() * 2 - 1
    factor_center_x = random.random() * 2 - 1
    factor_center_y = random.random() * 2 - 1

    if optimization == "parareal":
        factor_start_point_wave = 2
    else:
        factor_start_point_wave = 1

    if boundary_condition == "periodic":
        factor = 2 if optimization == "parareal" else 1
        center_x, center_y = factor_center_x * .45 / factor_start_point_wave / factor, factor_center_y * .45  / factor_start_point_wave / factor
        if res == 128:
            width = 1000 + factor_width*200
        elif res == 256:
            width = 7000 + factor_width*500
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    elif boundary_condition == "absorbing":
        center_x, center_y = ((factor_center_x * .45) / 2) / factor_start_point_wave, ((factor_center_y * .45) / 2) / factor_start_point_wave
        if res == 128:
            if optimization == "none":
                width = 5600 + factor_width * 500
            else:
                width = 20000 + factor_width * 800
        elif res == 256:
            width = 9000 + factor_width * 500
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    else: raise NotImplementedError("Boundary condition for initial condition not implemented.")

    return 2.0 / 128.0, width, center_x, center_y


def init_cond_ricker(xx, yy, width, center):
    """
    Ricker pulse wavefield
    """

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    u0 = (1 - 2 * width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * u0
    u0 = u0 / np.max(np.abs(u0))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


def get_velocities(n_it, res, boundary_condition, n_crops_other_profiles = 50, input_path = None, optimization = "none", prefix = ""):

    if input_path is None:
        # choose right bp_m dataset
        if boundary_condition == "absorbing":
            if optimization == "none":
                factor = 2
                if res == 128: input_path = prefix + '../data/velocity_profiles/crops_bp_m_400_128*2.npz'
                else: input_path = prefix + '../data/velocity_profiles/crops_bp_m_400_256*2.npz'
            else:
                factor = 1.5
                input_path = prefix + '../data/velocity_profiles/crops_bp_m_400_128*2.npz'
        else:
            factor = 1
            if res == 128: input_path = '../data/velocity_profiles/crops_bp_m_200_128.npz'
            else: input_path = '../data/velocity_profiles/crops_bp_m_400_128*2.npz'
    else:
        factor = 1
    # get velocities and save in dictionary
    res_padded = int(res * factor)
    velocities = get_velocity_dict(res_padded, n_crops_other_profiles, "../" + input_path, boundary_condition, optimization, prefix = prefix)

    # save velocities in tensor
    velocity_tensor = np.concatenate(list(velocities.values()), axis=0)

    # override number of iterations in data generation
    if n_it == -1:
        n_it = velocity_tensor.shape[0]
    else:
        idx = np.random.permutation(velocity_tensor.shape[0])[:n_it]
        velocity_tensor = velocity_tensor[idx]

    # get appendix for output path
    output_appendix = "__".join(velocities.keys())

    logging.info("Velocity models: " + output_appendix + "; "+str(velocity_tensor.shape))

    return velocity_tensor, n_it, res_padded, output_appendix


def get_velocity_dict(res_padded, n_crops_other_profiles, input_path, boundary_condition="absorbing", optimization = "none", prefix = ""):
    velocities = {
        # "diag": get_velocity_crop(res_padded, n_crops_other_profiles, "diagonal", boundary_condition, optimization, prefix),  # n_crops x res x res
        # "3l": get_velocity_crop(res_padded, n_crops_other_profiles, "three_layers", boundary_condition, optimization, prefix),  # n_crops x res x res
        # "cp": get_velocity_crop(res_padded, n_crops_other_profiles, "crack_profile", boundary_condition, optimization, prefix),  # n_crops x res x res
        # "hf": get_velocity_crop(res_padded, n_crops_other_profiles, "high_frequency", boundary_condition, optimization, prefix),  # n_crops x res x res
        "bp_m": np.load(input_path)['wavespeedlist']  # 200 x w_big x h_big
    }
    return velocities


def get_velocity_crop(resolution, n_crops, velocity_profile, boundary_conditon, optimization, prefix = ""):

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
        img = crack_profile(n_crops, resolution, boundary_conditon, optimization, prefix)

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


def crack_profile(n_it, res, boundary_condition, optimization, prefix):
    marmousi_datamat = loadmat(prefix + '../../data/velocity_profiles/marm1nonsmooth.mat')  # velocity models Marmousi dataset
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



# from generate_data.utils_wave import WaveEnergyField
# import matplotlib.pyplot as plt
# from generate_data.wave_propagation import pseudo_spectral_tensor, pseudo_spectral
#
# if __name__ == '__main__':
#     # "diag","3l","cp","hf", "bp_m
#
#     dx = 2./128.
#     dt = dx / 30.
#     Tf = .06
#
#     a = get_velocity_dict(128*3,1,input_path='../data/velocity_profiles/crops_bp_m_400_128*2.npz', optimization="parareal")
#     vel = a["diag"][0]
#
#     u,ut = initial_condition_gaussian(vel,128, "absorbing", 128*3, "parareal", mode="generate_data")
#
#     for s in range(11):
#         w = WaveEnergyField(u, ut, vel, dx)
#         plt.imshow(w[128:-128, 128:-128])
#         plt.show()
#         u,ut = pseudo_spectral(u,ut, vel, dx, dt, Tf)





