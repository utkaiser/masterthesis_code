import numpy as np
import matplotlib.pyplot as plt
import torch
from generate_data.wave_propagation import pseudo_spectral


def get_velocity_model(
        data_path,
        visualize = True
):
    '''

    Parameters
    ----------
    data_path : (string) path to velocity profile crops
    visualize : (boolean) whether to visualize data

    Returns
    -------
    (numpy array) single velocity profile
    '''

    # choose first velocity profile out of list of velocity crops
    vel = np.load(data_path)['wavespeedlist'].squeeze()[0]

    if visualize:
        plt.axis("off")
        plt.imshow(vel)
        plt.show()
        plt.title("Velocity profile")

    return vel


def pseudo_spectral_solutions(
        u,
        ut,
        vel,
        n_it,
        dx,
        dt,
        dt_star
):
    _,w,h = u.shape
    target = torch.zeros([n_it,2,w,h])
    target[0,0], target[0,1] = u, ut

    for i in range(n_it):
        target[i,0], target[i,1] = pseudo_spectral(target[i-1,0], target[i-1,1], vel, dx, dt, dt_star)

    return target



