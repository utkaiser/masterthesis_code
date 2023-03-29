import sys
sys.path.append("..")
sys.path.append("../..")
from os import path
import torch
from matplotlib import pyplot as plt
from parareal.generate_data.wave_util import WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor


def visualize_wavefield(tensor_list, dx = 2.0 / 128.0, f_delta_t=.06, scaler=2, vis_save=True, vis_path='../results/run_2'):
    # list of tupels with tensors

    n_snapshots = len(tensor_list)
    fig = plt.figure(figsize=(20, 8))
    loss_list = []
    f_delta_x = 2.0 / 128.0
    label_font_size = 5


    for i, values in enumerate(tensor_list):

        epoch, loss, input_idx, label_idx, input, output, label = values
        input, output, label = input.cpu(), output.cpu(), label.cpu()
        loss_list.append(str(round(loss,5)))

        combined_data = torch.stack([input[:3,:,:], output[:3,:,:], label])
        _min, _max = torch.min(combined_data), torch.max(combined_data)

        # velocity
        if i == 0:
            ax1 = fig.add_subplot(4, n_snapshots, i + 1)
            pos1 = ax1.imshow(input[3, :, :])
            plt.axis('off')
            c_v = plt.colorbar(pos1)
            c_v.ax.tick_params(labelsize=label_font_size)

        # input
        u_x, u_y, u_t_c, vel = input[0, :, :].unsqueeze(dim=0), input[1, :, :].unsqueeze(dim=0), input[2, :, :].unsqueeze(dim=0), input[3, :, :].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax2 = fig.add_subplot(4, n_snapshots, i+1 + n_snapshots)
        pos2 = ax2.imshow(WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax2.set_title('input', fontsize=10)
        c_i = plt.colorbar(pos2)
        c_i.ax.tick_params(labelsize=label_font_size)

        # output
        u_x, u_y, u_t_c = output[0, :, :].unsqueeze(dim=0), output[1, :, :].unsqueeze(dim=0), output[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax3 = fig.add_subplot(4, n_snapshots, i + 1 +n_snapshots*2)
        pos3 = ax3.imshow(WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax3.set_title('output', fontsize=10)
        c_o = plt.colorbar(pos3)
        c_o.ax.tick_params(labelsize=label_font_size)

        # label
        u_x, u_y, u_t_c = label[0, :, :].unsqueeze(dim=0), label[1, :, :].unsqueeze(dim=0), label[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax4 = fig.add_subplot(4, n_snapshots, i + 1 + n_snapshots*3)
        pos4 = ax4.imshow(WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax4.set_title('label', fontsize=10)
        c_l = plt.colorbar(pos4)
        c_l.ax.tick_params(labelsize=label_font_size)

    fig.suptitle("losses: " + ", ".join(loss_list), fontsize=14)

    if vis_save:
        for i in range(500):
            saving_path = vis_path + "epoch_" + str(epoch) + "_img_" + str(i) + ".png"
            if not path.isfile(saving_path):
                plt.savefig(saving_path)
                break
    else:
        plt.show()