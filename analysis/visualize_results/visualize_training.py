import matplotlib.pyplot as plt
from analysis.utils_analysis import get_ticks_fine, get_solver_solution
import torch
from models.model_utils import get_wavefield, round_loss, compute_loss


def visualize_wavefield(epoch, tensor_list, vel, vis_save, vis_path, initial_u):
    # initial_u -> b x c x w x h

    n_snaps = len(tensor_list)
    _, w, h = tensor_list[0][-1].shape
    fine_solver_tensor = torch.zeros([1,n_snaps, 3, w, h])

    for i, values in enumerate(tensor_list):
        _, _, label = values  # c x w x h
        fine_solver_tensor[0,i] = label.cpu()

    coarse_solver_tensor = get_solver_solution(initial_u[:, :3], n_snaps+1, initial_u[:, 3].unsqueeze(dim=0),
                                               solver="coarse")[1:]  # s x b x c x w x h

    ticks = get_ticks_fine(fine_solver_tensor, vel)  # s x 3

    fig = plt.figure(figsize=(20, 8))
    loss_list = []

    for i, values in enumerate(tensor_list):

        loss, output, label = values  # int, others: c x h x w
        output, label = output.cpu(), label.cpu()
        loss_list.append(loss)

        # velocity
        if i == 0:
            ax = fig.add_subplot(5, n_snaps, 1)
            pos = ax.imshow(vel)
            plt.colorbar(pos)
            ax.set_title("velocity", fontdict={'fontsize': 9})
            plt.axis('off')

        # velocity verlet
        ax = fig.add_subplot(5, n_snaps, n_snaps + 1 + i)
        wave_field = get_wavefield(coarse_solver_tensor[i], vel.unsqueeze(dim=0))
        pos = ax.imshow(wave_field)
        plt.colorbar(pos, ticks=ticks[i])
        ax.set_title(round_loss(compute_loss(coarse_solver_tensor[i], label.unsqueeze(dim=0), vel.unsqueeze(dim=0))),
                     fontdict={'fontsize': 9})
        plt.axis('off')

        # output
        ax = fig.add_subplot(5, n_snaps, n_snaps*2 + 1 + i)
        wave_field = get_wavefield(output.unsqueeze(dim=0), vel.unsqueeze(dim=0))
        pos = ax.imshow(wave_field, vmin=ticks[i][0], vmax=ticks[i][2])
        plt.colorbar(pos, ticks=ticks[i])
        ax.set_title(round_loss(compute_loss(output.unsqueeze(dim=0), label.unsqueeze(dim=0), vel.unsqueeze(dim=0))),
                         fontdict={'fontsize': 9})
        plt.axis('off')

        # label
        ax = fig.add_subplot(5, n_snaps, n_snaps*3 + 1 + i)
        wave_field = get_wavefield(label.unsqueeze(dim=0), vel.unsqueeze(dim=0))
        pos = ax.imshow(wave_field)
        plt.colorbar(pos, ticks=ticks[i])
        ax.set_title("label it " + str(i+1), fontdict={'fontsize': 9})
        plt.axis('off')

    fig.suptitle("Mean loss: " + str(sum(loss_list) / len(loss_list)), fontsize=14)

    if vis_save:
        saving_path = vis_path + "epoch_" + str(epoch) + ".png"
        plt.savefig(saving_path)
    else:
        plt.show()