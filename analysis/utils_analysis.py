import scipy
from scipy.io import savemat
import numpy as np


def change_npy_to_mat(dir_name = ""):
    # changes all files in folder from npy format to mat format

    a = np.load("vis_files/vel.npy")
    scipy.io.savemat('vel.mat', dict(res=a))
    # d = np.load("diagonal_fine.npy")
    # vel = get_velocity_crop(128, 1, "marmousi", "absorbing", "none")[0]


if __name__ == '__main__':
    change_npy_to_mat()