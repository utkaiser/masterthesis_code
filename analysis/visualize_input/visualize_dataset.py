from generate_data import utils_wave
from models.model_utils import fetch_data_end_to_end
import matplotlib.pyplot as plt
import torch
from generate_data.wave_propagation import velocity_verlet_tensor, pseudo_spectral


def vis_dataset():

    delta_t_star = .06
    dx = 2.0/128.0

    #get data
    path = "../../data/end_to_end_bp_m_10_2000.npz"
    loaders = fetch_data_end_to_end([path], shuffle=False, batch_size=1)

    for loader in loaders:
        for i, data in enumerate(loader):
            print(i, "-" * 150)

            input = data[0]  # b x n_snaps x c x w x h

            # velocity visualization
            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(input[0, 0, 3, :, :])
            plt.axis('off')
            plt.show()
            fig = plt.figure(figsize=(30, 5))

            # input vis
            idx = 0
            u_x, u_y, u_t_c, vel = input[:, idx, 0, :, :], input[:, idx, 1, :, :], input[:, idx, 2, :, :], input[:, idx,
                                                                                                           3, :, :]
            sumv = torch.sum(torch.sum(u_x))
            u, ut = utils_wave.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, dx, sumv)

            for idx in range(10):
                u, ut = velocity_verlet_tensor(u, ut, vel, dx, dx / 20, delta_t_star, number=1, boundary_c="absorbing")

                ax1 = fig.add_subplot(2, 10, idx + 1)
                pos1 = ax1.imshow(utils_wave.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)
                plt.colorbar(pos1)
                plt.axis('off')

                # #label vis
                # u_x, u_y, u_t_c, vel = input[:,idx+1, 0, :, :], input[:,idx+1, 1, :, :], input[:,idx+1, 2, :, :], input[:,idx+1, 3, :, :]
                # sumv = torch.sum(torch.sum(u_x))
                # u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, dx, sumv)
                #
                # ax2 = fig.add_subplot(2,10,idx+11)
                # pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u[0,:,:],ut[0,:,:],vel[0,:,:],dx)*dx*dx)
                # plt.colorbar(pos2)
                # plt.axis('off')

            plt.show()


if __name__ == '__main__':
    vis_dataset()



