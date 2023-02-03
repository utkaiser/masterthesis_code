from scipy.io import savemat
import numpy as np
import glob
import os
from generate_data.initial_conditions import get_velocity_crop
from generate_data.optimization.utils_optimization import smaller_crop, get_wavefield_numpy, get_wavefield


def change_npy_to_mat():
    # changes all files in folder from npy format to mat format
    npzFiles = glob.glob("*.npy")
    fm = os.path.splitext("diagonal_fine.npy")[0] + '.mat'
    d = np.load("diagonal_fine.npy")
    vel = get_velocity_crop(256, 1, "diagonal")[0]
    vel = smaller_crop(vel)
    d = get_wavefield_numpy(d[5, 0], vel)
    savemat(fm, {"res": d})


def get_ticks_fine(tensor, vel):
    # tensor -> 1 x s x c x w x h
    # vel -> w x h

    ticks = []
    for s in range(tensor.shape[1]):
        img = get_wavefield(tensor[:,s],vel)
        ticks.append([img.min().item(), (img.max().item() + img.min().item()) / 2 ,img.max().item()])
    return ticks  # s x 3