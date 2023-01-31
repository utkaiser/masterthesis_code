import matplotlib.pyplot as plt
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import torch

from models.optimization.utils_optimization import get_wavefield


# TODO: ticks same
# TODO: value range same
# TODO: loss well displayed


def visualize_wavefield(epoch, tensor_list, vis_save, vis_path):

    fig = plt.figure(figsize=(20, 8))
    loss_list = []
    label_font_size = 5
    n_snaps = len(tensor_list)

    c, w, h = tensor_list[0].shape
    vel = tensor_list[0][1][3]

    for i, values in enumerate(tensor_list):

        loss, inpt, output, label = values  # int, others: c x h x w
        loss_list.append(loss)

        # velocity
        if i == 0:
            ax = fig.add_subplot(4, n_snaps, 1)
            pos = ax.imshow(vel)
            plt.colorbar(pos)
            ax.set_title("velocity", fontdict={'fontsize': 9})
            plt.axis('off')

        for s in range(parareal_tensor.shape[1]):
            ax = fig.add_subplot(4, 11, 1 + s)
            wave_field = get_wavefield(coarse_solver_tensor[s], vel)
            pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
            if s != 0:
                plt.colorbar(pos, ticks=ticks[s])
                ax.set_title(round_loss(compute_loss(coarse_solver_tensor[s], fine_solver_tensor[s, :3], vel)),
                             fontdict={'fontsize': 9})
            plt.axis('off')



    fig.suptitle("Mean loss: " + str(sum(loss_list) / len(loss_list)), fontsize=14)

    if vis_save:
        saving_path = vis_path + "epoch_" + str(epoch) + ".png"
        plt.savefig(saving_path)
    else:
        plt.show()