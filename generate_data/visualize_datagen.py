import numpy as np
import matplotlib.pyplot as plt
from generate_data.change_wave_arguments import WaveEnergyComponentField_end_to_end, WaveSol_from_EnergyComponent,WaveEnergyField


def visualize_wavefield(u_elapse, ut_elapse, f_delta_x = 2.0 / 128.0, f_delta_t = (2.0 / 128.0) / 20., vel = None, frame=False, init_res_f=128, it=0, optimization="none"):

    init_res_f = 128
    wave_field = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)
    fig = plt.figure(figsize=(15, 8))
    u_x, u_y, u_t_c = wave_field
    # compute center
    axis_len = u_x.shape[-1]
    axis_center = axis_len // 2
    sumv = np.sum(np.sum(u_x))
    ax1 = fig.add_subplot(1,2,1)
    pos1 = ax1.imshow(vel)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)

    # plt.axis('off')

    u, ut = WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
    ax2 = fig.add_subplot(1,2, 2)

    pos2 = ax2.imshow(WaveEnergyField(u, ut.numpy(), vel.numpy(), f_delta_x) * f_delta_x * f_delta_x)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center + init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center + init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center + init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center - init_res_f // 2], 'r', linewidth=4)
    plt.plot([axis_center - init_res_f // 2, axis_center - init_res_f // 2],
             [axis_center - init_res_f // 2, axis_center + init_res_f // 2], 'r', linewidth=4)
    # plt.axis('off')


    plt.title("iteration: " + str(it))
    plt.show()

