import torch
import matplotlib.pyplot as plt
from matplotlib import cm

from generate_data.initial_conditions import initial_condition_gaussian, get_velocities
import sys
import numpy as np
from generate_data.utils_wave import WaveEnergyField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor
sys.path.append("..")

def vis_ps_batched():

    plt.figure(figsize=(30, 30))
    boundary_condition = "absorbing"
    res = 128
    dx, dt, Tf = 2./128., 1/600., .06
    velocity_tensor, _, _, _ = get_velocities(1, res, boundary_condition)
    vel = velocity_tensor.squeeze()
    np.save("../vis_files/vel.npy", vel)
    u, ut = initial_condition_gaussian(velocity_tensor, res=res, boundary_condition=boundary_condition, mode="generate_data",
                                       res_padded=128 * 2, optimization="parareal")

    u, ut, velocity_tensor = torch.from_numpy(u), torch.from_numpy(ut), torch.from_numpy(velocity_tensor)

    new_u, new_ut = torch.zeros([5,256,256]), torch.zeros([5,256,256])
    for b in range(5):
        new_u[b], new_ut[b] = u.clone(), ut.clone()

    for s in range(10):
        new_u, new_ut = pseudo_spectral_tensor(new_u, new_ut, velocity_tensor, dx, dt, Tf)
        w = WaveEnergyField_tensor(new_u[0], new_ut[0], velocity_tensor[0], dx)
        plt.imshow(vel,cmap='viridis')
        plt.imshow(w, alpha=0.8, cmap='viridis')
        plt.show()

if __name__ == '__main__':
    vis_ps_batched()