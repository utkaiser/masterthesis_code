import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from generate_data.wave_util import WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import torch

def round_loss(number):
    return number #str(round(number*(10**7),5))+"e-7"


def get_wavefield(tensor, vel, f_delta_x = 2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                           f_delta_x) * f_delta_x * f_delta_x


def plot_wavefield_results(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, MSE_loss, vel, vel_name, folder_name):
    fig = plt.figure(figsize=(35, 15))

    # coarse solver solution
    for s in range(parareal_tensor.shape[1]):
        ax = fig.add_subplot(7, 11, 1 + s)
        wave_field = get_wavefield(coarse_solver_tensor[s], vel)
        pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
        if s != 0:
            plt.colorbar(pos, ticks=ticks[s])
            ax.set_title(round_loss(MSE_loss(get_wavefield(fine_solver_tensor[s, :3], vel), wave_field).item()),
                         fontdict={'fontsize': 9})
        plt.axis('off')

    # parareal scheme
    for k in range(parareal_tensor.shape[0]):
        for s in range(parareal_tensor.shape[1]):
            wave_field = get_wavefield(parareal_tensor[k, s], vel)
            ax = fig.add_subplot(7, 11, 11 + 11 * k + s + 1)
            pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
            if s != 0:
                plt.colorbar(pos, ticks=ticks[s])
                ax.set_title(round_loss(MSE_loss(get_wavefield(fine_solver_tensor[s, :3], vel), wave_field).item()),
                             fontdict={'fontsize': 9})
            plt.axis('off')

    # fine solver solution
    for s in range(parareal_tensor.shape[1]):
        ax = fig.add_subplot(7, 11, 67 + s)
        wave_field = get_wavefield(fine_solver_tensor[s], vel)
        pos = ax.imshow(wave_field)
        if s != 0: plt.colorbar(pos, ticks=ticks[s])
        ax.set_title("fine solver it " + str(s), fontdict={'fontsize': 9})
        plt.axis('off')

    fig.suptitle("coarse solver (row 0), parareal end to end (k=0,...4) (row 1-5), fine solver (last row); titles represent MSE between result and fine solver")
    plt.savefig('../../results/parareal/'+folder_name+'/' + vel_name + '_wavefield_plot.pdf')
    plt.close(fig)


def get_ticks_fine(tensor, vel):
    # tensor -> s x b x c x w x h

    ticks = []
    for s in range(tensor.shape[0]):
        img = get_wavefield(tensor[s],vel)
        ticks.append([img.min().item(), (img.max().item() + img.min().item()) / 2 ,img.max().item()])
    return ticks  # s x 3