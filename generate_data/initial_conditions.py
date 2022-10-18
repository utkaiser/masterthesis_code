import numpy as np
import wave_propagation

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

def init_cond_gaussian(xx, yy, width, center):
    """
    Gaussian pulse wavefield
    """

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


def init_cond_ricker(xx, yy, width, center):
    """
    Ricker pulse wavefield
    """

    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    u0 = (1 - 2 * width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * u0
    u0 = u0 / np.max(np.abs(u0))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0