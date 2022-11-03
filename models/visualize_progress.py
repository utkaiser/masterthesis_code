import torch
from generate_data import wave_util
import matplotlib.pyplot as plt

def visualize_wavefield(tensor_list, dx = 2.0 / 128.0, f_delta_t=.2):
    # list of tupels with tensors

    #TODO: check why images are blank sometimes, check what you can do about it
    #TODO: implement way to show input_idx and so on, loss

    n_snapshots = len(tensor_list)
    fig = plt.figure(figsize=(30, 20))

    for i, values in enumerate(tensor_list):

        loss, input_idx, label_idx, input, output, label = values

        # input
        u_x, u_y, u_t_c, vel = input[0, :, :].unsqueeze(dim=0), input[1, :, :].unsqueeze(dim=0), input[2, :, :].unsqueeze(dim=0), input[3, :, :].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax1 = fig.add_subplot(3, n_snapshots, i+1)
        pos1 = ax1.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)
        plt.axis('off')

        # output
        u_x, u_y, u_t_c = output[0, :, :].unsqueeze(dim=0), output[1, :, :].unsqueeze(dim=0), output[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax2 = fig.add_subplot(3, n_snapshots, i + 1 +n_snapshots)
        pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)
        plt.axis('off')

        # label
        u_x, u_y, u_t_c = label[0, :, :].unsqueeze(dim=0), label[1, :, :].unsqueeze(dim=0), label[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax3 = fig.add_subplot(3, n_snapshots, i + 1 + n_snapshots*2)
        pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)
        plt.axis('off')

    plt.show()
