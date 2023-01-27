from scipy.io import savemat
import numpy as np
import glob
import os
from generate_data.initial_conditions import get_velocity_crop
from models.optimization.utils_optimization import smaller_crop, get_wavefield_numpy
import matplotlib.pyplot as plt

npzFiles = glob.glob("*.npy")
fm = os.path.splitext("diagonal_fine.npy")[0]+'.mat'
d = np.load("diagonal_fine.npy")
vel = get_velocity_crop(256,1,"diagonal")[0]
vel = smaller_crop(vel)
d = get_wavefield_numpy(d[5,0], vel)

savemat(fm, {"res" : d})