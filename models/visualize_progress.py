import torch
from generate_data import wave_util
import matplotlib.pyplot as plt
import numpy as np

def visualize_wavefield(tensor_list, dx = 2.0 / 128.0, f_delta_t=.06, scaler=2):
    # list of tupels with tensors

    n_snapshots = len(tensor_list)
    fig = plt.figure(figsize=(20, 8))
    loss_list = []

    for i, values in enumerate(tensor_list):

        loss, input_idx, label_idx, input, output, label = values
        loss_list.append(str(round(loss,5)))

        combined_data = torch.stack([input[:3,:,:], output[:3,:,:], label])
        _min, _max = torch.min(combined_data), torch.max(combined_data)

        # velocity
        if i == 0:
            ax1 = fig.add_subplot(4, n_snapshots, i + 1)
            pos1 = ax1.imshow(input[3, :, :])
            plt.axis('off')
            #plt.colorbar(pos1)

        # input
        u_x, u_y, u_t_c, vel = input[0, :, :].unsqueeze(dim=0), input[1, :, :].unsqueeze(dim=0), input[2, :, :].unsqueeze(dim=0), input[3, :, :].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax2 = fig.add_subplot(4, n_snapshots, i+1 + n_snapshots)
        pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        #ax2.set_title('loss: '+str(round(loss,5))+", input_idx: "+str(input_idx)+", label_idx: "+str(label_idx), fontsize=20)
        # plt.colorbar(pos2)

        # output
        u_x, u_y, u_t_c = output[0, :, :].unsqueeze(dim=0), output[1, :, :].unsqueeze(dim=0), output[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax3 = fig.add_subplot(4, n_snapshots, i + 1 +n_snapshots*2)
        pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        # plt.colorbar(pos3)

        # label
        u_x, u_y, u_t_c = label[0, :, :].unsqueeze(dim=0), label[1, :, :].unsqueeze(dim=0), label[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax4 = fig.add_subplot(4, n_snapshots, i + 1 + n_snapshots*3)
        pos4 = ax4.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        # plt.colorbar(pos4)

    fig.suptitle("losses: " + ", ".join(loss_list), fontsize=14)
    plt.show()
