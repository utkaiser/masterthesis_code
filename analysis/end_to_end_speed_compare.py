import sys
sys.path.append("..")
from generate_data import wave_util
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import model_end_to_end
from skimage.transform import resize
from models.model_utils import fetch_data_end_to_end, get_params
from generate_data.wave_propagation import velocity_verlet_tensor, pseudo_spectral, velocity_verlet

def compare_end_to_end_gpu():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())

    # params
    boundary_c = 'absorbing'
    dx = 2.0 / 128.0
    dt = dx / 20
    dX = 2./64.
    dT = 1./500.
    delta_t_star = .06
    scaler = 2
    n_snaps = 12
    Nx, Ny = 128, 128
    c_Nx, c_Ny = 64, 64

    path = ['../data/end_to_end_bp_m_10_2000.npz']
    loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=False, batch_size=1)

    # set up models
    param_dict = get_params("0")
    restr_model1 = model_end_to_end.Restriction_nn(param_dict = param_dict).double().to(device)
    # restr_model1 = torch.nn.DataParallel(restr_model1)
    # restr_model1.load_state_dict(torch.load('../results/run_2/saved_model_end_to_end_unet128_29.pt'))
    # restr_model1.eval()

    netlist = [
        (r'end-to-end unet 3lvl', restr_model1),
    ]

    for netname, netmodl in netlist:
        model_parameters = filter(lambda p: p.requires_grad, netmodl.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(netname, 'number of trainable parameters', params)

    uc, utc = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
    uf, utf = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
    uo, uto = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
    e_time_fine, s_time_fine = [], []
    e_time_coarse, s_time_coarse = [], []
    e_time_endtoend, s_time_endtoend = [], []

    with torch.no_grad():
        for i, data in enumerate(loader):

            input = data[0].squeeze()  # n_snaps x 4 x w x h
            fig = plt.figure(figsize=(35, 8))

            # initial condition visualization
            u_x, u_y, u_t_c, vel = input[0, 0, :, :], input[0, 1, :, :], input[0, 2, :, :], input[0, 3, :, :]
            sumv = torch.sum(torch.sum(u_x))
            u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x.unsqueeze(dim=0), u_y.unsqueeze(dim=0),
                                                                  u_t_c.unsqueeze(dim=0), vel.unsqueeze(dim=0), dx,
                                                                  sumv)

            ax2 = fig.add_subplot(3, 11, 1)
            pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), vel, dx) * dx * dx)
            ax2.set_title('init condition', fontsize=10)
            plt.colorbar(pos2)
            plt.axis('off')

            # velocity visualization
            ax1 = fig.add_subplot(3, 11, 12)
            vel_img = input[0, 3, :, :]
            pos1 = ax1.imshow(vel_img)
            plt.colorbar(pos1)
            plt.axis('off')

            # # fine solver iteration
            # for j in range(1, input.shape[0]):
            #     u_x, u_y, u_t_c, vel = input[j, 0, :, :], input[j, 1, :, :], input[j, 2, :, :], input[j, 3, :,
            #                                                                                     :]  # w x h
            #     sumv = torch.sum(torch.sum(u_x))
            #     u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x.unsqueeze(dim=0), u_y.unsqueeze(dim=0),
            #                                                           u_t_c.unsqueeze(dim=0), vel.unsqueeze(dim=0),
            #                                                           dx, sumv)
            #     uf[j - 1, :, :], utf[j - 1, :, :] = u.squeeze(), ut.squeeze()
            #     ax = fig.add_subplot(3, 12, 1 + j)
            #     pos = ax.imshow(wave_util.WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), vel, dx) * dx * dx)
            #     ax.set_title('it' + str(j), fontsize=10)
            #     plt.colorbar(pos)
            #     plt.axis('off')

            ufx, ufcx = wave_util.WaveSol_from_EnergyComponent_tensor(input[0, 0, :, :].unsqueeze(dim=0),
                                                                      input[0, 1, :, :].unsqueeze(dim=0),
                                                                      input[0, 2, :, :].unsqueeze(dim=0),
                                                                      input[0, 3, :, :].unsqueeze(dim=0), dx, sumv)
            for j in range(1, input.shape[0]):
                s_time_fine.append(time.process_time())
                ufx, ufcx = velocity_verlet_tensor(ufx, ufcx, vel.unsqueeze(dim=0), dx, dt, delta_t_star, number=1, boundary_c="absorbing") #number=1,
                                                   #boundary_c=boundary_c)  # pseudo_spectral
                e_time_fine.append(time.process_time())
                ax = fig.add_subplot(3,11,1+j)
                pos = ax.imshow(wave_util.WaveEnergyField_tensor(ufx.squeeze(),ufcx.squeeze(),vel,dx)*dx*dx)
                plt.colorbar(pos)
                plt.axis('off')

            # coarse solver
            ucx, utcx = wave_util.WaveSol_from_EnergyComponent_tensor(input[0, 0, :, :].unsqueeze(dim=0),
                                                                      input[0, 1, :, :].unsqueeze(dim=0),
                                                                      input[0, 2, :, :].unsqueeze(dim=0),
                                                                      input[0, 3, :, :].unsqueeze(dim=0), dx, sumv)
            ucx, utcx = ucx.squeeze(), utcx.squeeze()
            for j in range(1, input.shape[0]):
                ucx, utcx, vel_c = torch.from_numpy(resize(ucx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(
                    resize(utcx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(resize(vel.cpu(), [c_Nx, c_Nx], order=4))
                ucx, utcx, vel_c = ucx.unsqueeze(dim=0).to(device), utcx.unsqueeze(dim=0).to(device), vel_c.unsqueeze(dim=0).to(device)
                s_time_coarse.append(time.process_time())
                ucx, utcx = velocity_verlet_tensor(
                    ucx, utcx,
                    vel_c, dX, dT, delta_t_star, number=1, boundary_c=boundary_c
                )
                e_time_coarse.append(time.process_time())
                ucx, utcx = ucx.squeeze(), utcx.squeeze()
                ucx, utcx = torch.from_numpy(resize(ucx.cpu(), [Nx, Nx], order=4)), torch.from_numpy(
                    resize(utcx.cpu(), [Nx, Nx], order=4))
                uc[j - 1, :, :], utc[j - 1, :, :] = ucx, utcx
                ax2 = fig.add_subplot(3, 12, 13 + j)
                pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(ucx, utcx, vel.cpu(), dx) * dx * dx)
                plt.colorbar(pos2)
                plt.axis('off')

            # restriction
            input_restr = input[0, :3, :, :].unsqueeze(dim=0)
            for j in range(1, input.shape[0]):
                input_restr = torch.concat([input_restr.cpu(), vel_img.unsqueeze(dim=0).unsqueeze(0).cpu()], dim=1)
                input_restr = input_restr.to(device)
                vel = vel.to(device)
                s_time_endtoend.append(time.process_time())
                output = restr_model1(input_restr)  # b x 3 x w x h
                e_time_endtoend.append(time.process_time())
                u_x, u_y, u_t_c = output[:, 0, :, :], output[:, 1, :, :], output[:, 2, :, :]
                sumv = torch.sum(torch.sum(u_x))
                u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, dt, sumv)
                uo[j - 1, :, :], uto[j - 1, :, :] = u.squeeze().cpu(), ut.squeeze().cpu()
                ax3 = fig.add_subplot(3, 12, 25 + j)
                pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u.squeeze().cpu(), ut.squeeze().cpu(), vel.cpu(), dx) * dx * dx)
                plt.colorbar(pos3)
                plt.axis('off')
                input_restr = output.clone()

            break


    plt.savefig('../results/run_2/test.png')

    # time difference
    print("time coarse solver:", np.subtract(np.array(e_time_coarse),np.array(s_time_coarse)).mean())
    print("time fine solver:", np.subtract(np.array(e_time_fine),np.array(s_time_fine)).mean())
    print("time end to end solver:", np.subtract(np.array(e_time_endtoend), np.array(s_time_endtoend)).mean())

if __name__ == '__main__':
    compare_end_to_end_gpu()





























# import sys
# sys.path.append("..")
# from generate_data import wave_util
# import time
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from models import model_end_to_end
# from skimage.transform import resize
# import torch.nn.functional as F
# from models.model_utils import fetch_data_end_to_end, get_params
# from generate_data.wave_propagation import velocity_verlet_tensor, pseudo_spectral, velocity_verlet
#
# def compare_end_to_end_gpu():
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
#
#     # params
#     boundary_c = 'absorbing'
#     dx = 2.0 / 128.0
#     dt = dx / 20
#     dX = 2./64.
#     dT = 1./500.
#     delta_t_star = .06
#     scaler = 2
#     n_snaps = 12
#     Nx, Ny = 128, 128
#     c_Nx, c_Ny = 64, 64
#
#     path = ['../data/end_to_end_bp_m_10_2000.npz']
#     loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=False, batch_size=1)
#
#     # set up models
#     param_dict = get_params("0")
#     restr_model1 = model_end_to_end.Restriction_nn(param_dict = param_dict).double().to(device)
#     # restr_model1 = torch.nn.DataParallel(restr_model1)
#     # restr_model1.load_state_dict(torch.load('../results/run_2/saved_model_end_to_end_unet128_29.pt'))
#     # restr_model1.eval()
#
#     netlist = [
#         (r'end-to-end unet 3lvl', restr_model1),
#     ]
#
#     for netname, netmodl in netlist:
#         model_parameters = filter(lambda p: p.requires_grad, netmodl.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         print(netname, 'number of trainable parameters', params)
#
#     uc, utc = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
#     uf, utf = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
#     uo, uto = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
#     e_time_fine, s_time_fine = [], []
#     e_time_coarse, s_time_coarse = [], []
#     e_time_endtoend, s_time_endtoend = [], []
#
#     x = np.arange(-1, 1, dx)
#     y = np.arange(-1, 1, dx)
#     xx, yy = np.meshgrid(x, y)
#
#     vel = torch.from_numpy(1. + 0.0 * yy - 0.5 * (np.abs(yy + xx - 0.) > 0.4) + 0. * (np.abs(xx - 0.4) < 0.2) * (np.abs(yy - 0.5) < 0.1))
#     # vel = input[0, 3, :, :]#torch.from_numpy()
#     center = np.array([0., 0.])
#     u0 = torch.from_numpy(np.exp(-250.0 * (0.2 * (xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * np.cos(
#         8 * np.pi * (yy - center[1])))
#     ut0 = torch.from_numpy(np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)]))
#
#     with torch.no_grad():
#         for i, data in enumerate(loader):
#
#             input = data[0].squeeze()  # n_snaps x 4 x w x h
#             #
#             #
#             fig = plt.figure(figsize=(35, 8))
#             #
#             # # initial condition visualization
#             # u_x, u_y, u_t_c, vel = input[0, 0, :, :], input[0, 1, :, :], input[0, 2, :, :], input[0, 3, :, :]
#             # sumv = torch.sum(torch.sum(u_x))
#             # u, ut = u0, ut0 #wave_util.WaveSol_from_EnergyComponent_tensor(u_x.unsqueeze(dim=0), u_y.unsqueeze(dim=0),
#             #                  #                                     u_t_c.unsqueeze(dim=0), vel.unsqueeze(dim=0), dx,
#             #                   #                                    sumv)
#             #
#             # ax2 = fig.add_subplot(3, 11, 1)
#             # pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), vel, dx) * dx * dx)
#             # ax2.set_title('init condition', fontsize=10)
#             # plt.colorbar(pos2)
#             # plt.axis('off')
#             #
#             # # velocity visualization
#             # ax1 = fig.add_subplot(3, 11, 12)
#             # vel_img = vel
#             # pos1 = ax1.imshow(vel_img)
#             # plt.colorbar(pos1)
#             # plt.axis('off')
#
#             # fine solver iteration
#             # for j in range(1, input.shape[0]):
#             #     u_x, u_y, u_t_c, vel = input[j, 0, :, :], input[j, 1, :, :], input[j, 2, :, :], input[j, 3, :,
#             #                                                                                     :]  # w x h
#             #     sumv = torch.sum(torch.sum(u_x))
#             #     u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x.unsqueeze(dim=0), u_y.unsqueeze(dim=0),
#             #                                                           u_t_c.unsqueeze(dim=0), vel.unsqueeze(dim=0),
#             #                                                           dx, sumv)
#             #     uf[j - 1, :, :], utf[j - 1, :, :] = u.squeeze(), ut.squeeze()
#             #     ax = fig.add_subplot(3, 12, 1 + j)
#             #     pos = ax.imshow(wave_util.WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), vel, dx) * dx * dx)
#             #     ax.set_title('it' + str(j), fontsize=10)
#             #     plt.colorbar(pos)
#             #     plt.axis('off')
#
#             ufx, ufcx = wave_util.WaveSol_from_EnergyComponent_tensor(input[0, 0, :, :].unsqueeze(dim=0),
#                                                                       input[0, 1, :, :].unsqueeze(dim=0),
#                                                                       input[0, 2, :, :].unsqueeze(dim=0),
#                                                                       input[0, 3, :, :].unsqueeze(dim=0), dx, torch.sum(torch.sum(input[0, 0, :, :].unsqueeze(dim=0))))
#
#             vel = input[0, 3, :, :]
#             #ufx, ufcx = u0.clone().squeeze(), ut0.clone().squeeze()
#             # ufx, ufcx, vel_c = torch.from_numpy(resize(ufx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(
#             #     resize(ufcx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(resize(vel.cpu(), [c_Nx, c_Nx], order=4))
#
#             # coarse solver
#             ufx, ufcx = ufx.squeeze(), ufcx.squeeze()
#             for j in range(1, input.shape[0]):
#                 ufx, ufcx, vel_c = torch.from_numpy(resize(ufx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(
#                     resize(ufcx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(
#                     resize(vel.cpu(), [c_Nx, c_Nx], order=4))
#                 ufx, ufcx, vel_c = ufx.unsqueeze(dim=0).to(device), ufcx.unsqueeze(dim=0).to(device), vel_c.unsqueeze(
#                     dim=0).to(device)
#                 s_time_coarse.append(time.process_time())
#                 ufx, ufcx = velocity_verlet_tensor(
#                     ufx, ufcx,
#                     vel_c, dX, dT, delta_t_star, number=1, boundary_c="absorbing"
#                 )
#                 e_time_coarse.append(time.process_time())
#                 ufx, ufcx = ufx.squeeze(), ufcx.squeeze()
#                 ufx, ufcx = torch.from_numpy(resize(ufx.cpu(), [Nx, Nx], order=4)), torch.from_numpy(
#                     resize(ufcx.cpu(), [Nx, Nx], order=4))
#                 ax2 = fig.add_subplot(3, 12, 13 + j)
#                 pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(ufx, ufcx, vel.cpu(), dx) * dx * dx)
#                 plt.colorbar(pos2)
#                 plt.axis('off')
#
#
#             # coarse solver
#             # ucx, utcx = wave_util.WaveSol_from_EnergyComponent_tensor(input[0, 0, :, :].unsqueeze(dim=0),
#             #                                                           input[0, 1, :, :].unsqueeze(dim=0),
#             #                                                           input[0, 2, :, :].unsqueeze(dim=0),
#             #                                                           input[0, 2, :, :].unsqueeze(dim=0),
#             #                                                           dx,
#             #                                                           torch.sum(torch.sum(input[0, 0, :, :].unsqueeze(dim=0))))
#
#             ucx, utcx = u0.clone().unsqueeze(dim=0), ut0.clone().unsqueeze(dim=0)
#
#             for j in range(1, input.shape[0]-1):
#                 s_time_coarse.append(time.process_time())
#                 ucx, utcx = velocity_verlet_tensor(
#                     ucx, utcx,
#                     vel.unsqueeze(dim=0), dx, dt, delta_t_star, number=1, boundary_c=boundary_c
#                 )
#                 e_time_coarse.append(time.process_time())
#
#                 ax2 = fig.add_subplot(2, 11, 12 + j)
#                 pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(ucx.squeeze(), utcx.squeeze(), vel.cpu().squeeze(), dx) * dx * dx)
#                 ax2.set_title("fine solver")
#                 plt.colorbar(pos2)
#                 plt.axis('off')
#
#
#
#             # # restriction
#             # input_restr = input[0, :3, :, :].unsqueeze(dim=0)
#             # for j in range(1, input.shape[0]):
#             #     input_restr = torch.concat([input_restr.cpu(), vel_img.unsqueeze(dim=0).unsqueeze(0).cpu()], dim=1)
#             #     input_restr = input_restr.to(device)
#             #     vel = vel.to(device)
#             #     s_time_endtoend.append(time.process_time())
#             #     output = restr_model1(input_restr)  # b x 3 x w x h
#             #     e_time_endtoend.append(time.process_time())
#             #     u_x, u_y, u_t_c = output[:, 0, :, :], output[:, 1, :, :], output[:, 2, :, :]
#             #     sumv = torch.sum(torch.sum(u_x))
#             #     u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, dt, sumv)
#             #     uo[j - 1, :, :], uto[j - 1, :, :] = u.squeeze().cpu(), ut.squeeze().cpu()
#             #     ax3 = fig.add_subplot(3, 12, 25 + j)
#             #     pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u.squeeze().cpu(), ut.squeeze().cpu(), vel.cpu(), dx) * dx * dx)
#             #     plt.colorbar(pos3)
#             #     plt.axis('off')
#             #     input_restr = output.clone()
#
#             break
#
#
#     plt.savefig('../results/run_2/test.png')
#
#     # time difference
#     print("time coarse solver:", np.subtract(np.array(e_time_coarse),np.array(s_time_coarse)).mean())
#     print("time fine solver:", np.subtract(np.array(e_time_fine),np.array(s_time_fine)).mean())
#     print("time end to end solver:", np.subtract(np.array(e_time_endtoend), np.array(s_time_endtoend)).mean())
#
# if __name__ == '__main__':
#     compare_end_to_end_gpu()
