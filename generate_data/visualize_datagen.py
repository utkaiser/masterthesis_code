import numpy as np
import matplotlib.pyplot as plt
from generate_data.change_wave_arguments import WaveEnergyComponentField_end_to_end, WaveSol_from_EnergyComponent,WaveEnergyField
from generate_data.utils import get_wavefield


def visualize_wavefield(
        u_elapse,
        ut_elapse,
        f_delta_x = 2.0 / 128.0,
        f_delta_t = (2.0 / 128.0) / 20.,
        vel = None,
        it=0
):
    '''
    Parameters
    ----------
    u_elapse : (pytorch tensor) physical wave component, displacement of wave
    ut_elapse : (pytorch tensor) physical wave component derived by t, velocity of wave
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size
    vel : (pytorch tensor) velocity profile
    it : number of iteration, used when printing results

    Returns
    -------
    visualize wave as energy-semi norm when generating the data
    '''

    init_res_f = 128
    wave_field = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)
    fig = plt.figure(figsize=(15, 8))
    u_x, u_y, u_t_c = wave_field

    # compute center
    axis_len = u_x.shape[-1]
    axis_center = axis_len // 2
    sumv = np.sum(np.sum(u_x))
    ax1 = fig.add_subplot(1,2,1)
    _ = ax1.imshow(vel)

    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)

    u, ut = WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
    ax2 = fig.add_subplot(1,2, 2)

    __ = ax2.imshow(WaveEnergyField(u, ut.numpy(), vel.numpy(), f_delta_x) * f_delta_x * f_delta_x)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)

    plt.title("iteration: " + str(it))
    plt.show()



def visualize_wavefield_old_paper(
        u_n_coarse,
        u_n,
        c_delta_x,
        c_delta_t,
        f_delta_x,
        f_delta_t,
        vel,
        vel_crop,
        vel_crop_coarse,
        it = 0,
):
    '''
    Parameters
    ----------
    u_elapse : (pytorch tensor) physical wave component, displacement of wave
    ut_elapse : (pytorch tensor) physical wave component derived by t, velocity of wave
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size
    vel : (pytorch tensor) velocity profile
    it : number of iteration, used when printing results

    Returns
    -------
    visualize wave as energy-semi norm when generating the data
    '''

    fig = plt.figure(figsize=(15, 8))
    init_res_f = 128
    axis_center = 128

    wavefield_coarse = get_wavefield(u_n_coarse, vel_crop_coarse, c_delta_x, c_delta_t)
    wavefield = get_wavefield(u_n, vel_crop, f_delta_x, f_delta_t)

    ax1 = fig.add_subplot(1, 3, 1)
    _ = ax1.imshow(vel)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)

    ax2 = fig.add_subplot(1, 3, 2)
    _ = ax2.imshow(wavefield * f_delta_x * f_delta_x)

    ax3 = fig.add_subplot(1, 3, 3)
    _ = ax3.imshow(wavefield_coarse * c_delta_x * c_delta_x)

    plt.title("iteration: " + str(it))
    plt.show()