import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from generate_data.utils import get_wavefield, smaller_crop
from generate_data.utils_wave_propagate import one_iteration_velocity_verlet_tensor, one_iteration_pseudo_spectral_tensor
from models.utils import round_loss, compute_loss


def visualize_wavefield(
        epoch,
        tensor_list,
        vel,
        vis_save,
        vis_path,
        initial_u
):
    '''
    Parameters
    ----------
    epoch : (int) number of epochs to train
    tensor_list : (list of pytorch tensors) list of elements to plot
    vel :  (pytorch tensor) velocity profile
    vis_save : (bool) decides if visualization is saved
    vis_path : (string) path to save visualization
    initial_u : (pytorch tensor) wave representation as energy components

    Returns
    -------
    visualize training epoch
    '''

    n_snaps = len(tensor_list)
    _, w, h = tensor_list[0][-1].shape
    fine_solver_tensor = torch.zeros([1,n_snaps, 3, w, h])

    for i, values in enumerate(tensor_list):
        _, _, label = values  # c x w x h
        fine_solver_tensor[0,i] = label.cpu()

    coarse_solver_tensor = get_solver_solution(initial_u[:, :3], n_snaps+1, initial_u[:, 3].unsqueeze(dim=0),
                                               solver="coarse")[:,1:]  # s x b x c x w x h
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
        wave_field = get_wavefield(coarse_solver_tensor[:,i], vel.unsqueeze(dim=0))
        pos = ax.imshow(wave_field)
        plt.colorbar(pos, ticks=ticks[i])
        ax.set_title(round_loss(compute_loss(coarse_solver_tensor[:,i], label.unsqueeze(dim=0), vel.unsqueeze(dim=0))),
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


def get_ticks_fine(
        tensor,
        vel
):
    '''
    Parameters
    ----------
    tensor : (pytorch tensor) matrix
    vel : (pytorch tensor) velocity profile

    Returns
    -------
    (list) get ticks for visualization; usual values: [min, avg, max}
    '''

    ticks = []
    for s in range(tensor.shape[1]):
        img = get_wavefield(tensor[:,s],vel)
        ticks.append([img.min().item(), (img.max().item() + img.min().item()) / 2 ,img.max().item()])
    return ticks  # s x 3


def get_solver_solution(
        u_n_k,
        n_snapshots,
        vel,
        solver="coarse"
):
    '''
    Parameters
    ----------
    u_n_k :  (pytorch tensor) wave representation as energy components
    n_snapshots : (int) number of wave advancements by end-to-end model
    vel : (pytorch tensor) velocity profile
    solver : (string) choice of solver

    Returns
    -------
    compute solution of {solver} with down and up samplign to compare it with end-to-end model
    '''

    if solver == "coarse":
        small_res_scale = 2
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([b, n_snapshots, c, w, h])
        for s in range(n_snapshots):
            sol[0, s] = u_n_k.squeeze()

            a = F.upsample(u_n_k[:, 0].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale),
                           mode='bilinear')
            b1 = F.upsample(u_n_k[:, 1].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale),
                            mode='bilinear')
            b2 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale),
                            mode='bilinear')
            d = F.upsample(vel, size=(w // small_res_scale, w // small_res_scale), mode='bilinear')

            u_n_k = torch.concat([a, b1, b2, d], dim=1)

            u_n_k = one_iteration_velocity_verlet_tensor(u_n_k, c_delta_x=2. / 64., c_delta_t=1. / 600., delta_t_star=.06)

            a2 = F.upsample(u_n_k[:, 0].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 1].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b22 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w, w), mode='bilinear')

            u_n_k = torch.concat([a2, b2, b22], dim=1)

    elif solver == "fine":
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w // 2, h // 2])

        for s in range(n_snapshots):
            sol[s, :] = smaller_crop(u_n_k)
            u_n_k = torch.concat([u_n_k, vel], dim=1)
            u_n_k = one_iteration_pseudo_spectral_tensor(u_n_k)

    else:
        raise NotImplementedError("This solver has not been implemented yet.")

    return sol