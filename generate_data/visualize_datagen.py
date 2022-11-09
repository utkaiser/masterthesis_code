import numpy as np
from generate_data import wave_util
import matplotlib.pyplot as plt

def visualize_wavefield(tensors, dx = 2.0 / 128.0, f_delta_t=.2, vel = None, frame=False, init_res_f=128):

    f_delta_t = .1

    fig = plt.figure(figsize=(30, 20))
    u_x, u_y, u_t_c = tensors
    sumv = np.sum(np.sum(u_x))
    ax1 = fig.add_subplot(1,2, 1)
    pos1 = ax1.imshow(vel)
    plt.axis('off')

    # compute center
    axis_len = u_x.shape[-1]
    axis_center = axis_len // 2

    if frame:
        u, ut = wave_util.WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax2 = fig.add_subplot(1,2, 2)
        pos2 = ax2.imshow(wave_util.WaveEnergyField(u, ut, vel, dx) * dx * dx)
        plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
                 [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
        plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
                 [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
        plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
                 [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
        plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
                 [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)

    plt.show()
