import numpy as np
from generate_data import wave_propagation

def first_guess_integration(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star, n_snapshots, n_parareal_it,
                  resolution_f):

    # store solution at every time slice and parareal iteration
    up = np.zeros([resolution_f, resolution_f, n_snapshots, n_parareal_it])
    utp = np.zeros([resolution_f, resolution_f, n_snapshots, n_parareal_it])

    # set initial condition
    up[:, :, 0, :] = np.repeat(u_elapse[:, :, np.newaxis], n_parareal_it, axis=2)
    utp[:, :, 0, :] = np.repeat(ut_elapse[:, :, np.newaxis], n_parareal_it, axis=2)

    # first integration; initial guess just propagating using fine solver
    for j in range(1, n_snapshots):
        u_elapse, ut_elapse = wave_propagation.velocity_verlet(u_elapse, ut_elapse, vel,
                                                               f_delta_x, f_delta_t, delta_t_star)
        up[:, :, j, 0], utp[:, :, j, 0] = u_elapse, ut_elapse

    return up, utp

def init_cond_gaussian_old(xx, yy, width, center):
    """
    Gaussian pulse wavefield
    """

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


def init_cond_gaussian(it, init_res_f, res_f, absorbing_bc=True):
    """
    Gaussian pulse wavefield
    """

    centers, widths = np.random.rand(1, 2) * 1. - 0.5, 250 + np.random.randn(1) * 10

    if absorbing_bc:
        res_f = 300  # math.ceil((init_res_f + math.ceil(np.amax(vel) * delta_t_star * (n_snaps+1) * init_res_f) + 5) / 2.) * 2 # max value the wave can propagate for delta t star, plus rounding errors into account

    widths_scaler = 3 + (res_f ** 1.28 / 1000) if absorbing_bc else 1
    curr_centers, curr_widths = centers / (res_f / init_res_f), widths * widths_scaler * (
                res_f / init_res_f)  # scale init condition
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res_f), np.linspace(-1, 1, res_f))

    xx = grid_x
    yy = grid_y
    width = curr_widths[0]
    center = curr_centers[0]

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])

    return u0, ut0, res_f


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

def diagonal_ray(n_it, res = 300):

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    vel_profile = torch.from_numpy(3. + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.) > 0.2))
    return vel_profile.unsqueeze(dim=0).repeat(n_it,1,1).numpy()
