import sys
sys.path.append("")
import matplotlib.pyplot as plt
import torch


def plot_wavefield_results(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, MSE_loss, vel, vel_name):
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
    plt.savefig('../../results/parareal/check_stability/' + vel_name + '_wavefield_plot.pdf')
    plt.close(fig)