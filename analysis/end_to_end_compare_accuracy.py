import sys
sys.path.append("..")
from generate_data import wave_util
import numpy as np
import matplotlib.pyplot as plt
from models import model_end_to_end
from skimage.transform import resize
from models.model_utils import fetch_data_end_to_end, get_params
from generate_data.wave_propagation import velocity_verlet_tensor, pseudo_spectral, velocity_verlet
import torch

def compare_end_to_end_gpu():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())

    # params
    param_dict = get_params("0")

    boundary_c = param_dict["boundary_c"]
    dx = param_dict["f_delta_x"]
    dt = param_dict["f_delta_t"]
    dX = param_dict["c_delta_x"]
    dT = param_dict["c_delta_t"]
    delta_t_star = param_dict["delta_t_star"]
    scaler = param_dict["res_scaler"]
    n_snaps = 10
    Nx, Ny = 128, 128
    c_Nx, c_Ny = 64, 64

    path = ['../data/end_to_end_bp_m_10_2000.npz']
    loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=True, batch_size=1)

    # set up models
    restr_model = model_end_to_end.Restriction_nn(param_dict = param_dict).double().to(device)
    restr_model = torch.nn.DataParallel(restr_model)
    restr_model.load_state_dict(torch.load('../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10.pt'))
    restr_model.eval()

    netlist = [
        (r'end-to-end unet 3lvl', restr_model),
    ]

    for netname, netmodl in netlist:
        model_parameters = filter(lambda p: p.requires_grad, netmodl.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(netname, 'number of trainable parameters', params)

    # uc, utc = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
    # uf, utf = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])
    # uo, uto = np.zeros([n_snaps, Nx, Ny]), np.zeros([n_snaps, Nx, Ny])

    c_solver_acc_list = []
    model_acc_list = []
    f_solver_res = []
    loss_f = torch.nn.MSELoss()

    with torch.no_grad():
        for i, data in enumerate(loader):

            # initial condition visualization
            input = data[0].squeeze()  # n_snaps x 4 x w x h
            u_x, u_y, u_t_c, vel = input[0, 0, :, :], input[0, 1, :, :], input[0, 2, :, :], input[0, 3, :, :]
            sumv = torch.sum(torch.sum(u_x))

            # fine solver results
            for j in range(1,n_snaps):
                a, b, c, d = input[j, 0, :, :], input[j, 1, :, :], input[j, 2, :, :], input[j, 3, :, :]
                sumv2 = torch.sum(torch.sum(a))
                e, f = wave_util.WaveSol_from_EnergyComponent_tensor(u_x.unsqueeze(dim=0), u_y.unsqueeze(dim=0),
                                                                     u_t_c.unsqueeze(dim=0), vel.unsqueeze(dim=0),
                                                                     dx, sumv2)
                g = wave_util.WaveEnergyField_tensor(e.squeeze(), f.squeeze(), vel.squeeze(), dx) * dx * dx
                f_solver_res.append(g)




            # coarse solver
            ucx, utcx = wave_util.WaveSol_from_EnergyComponent_tensor(input[0, 0, :, :].unsqueeze(dim=0),
                                                                      input[0, 1, :, :].unsqueeze(dim=0),
                                                                      input[0, 2, :, :].unsqueeze(dim=0),
                                                                      input[0, 3, :, :].unsqueeze(dim=0), dx, sumv)
            ucx, utcx = ucx.squeeze(), utcx.squeeze()
            for j in range(1, n_snaps):
                ucx, utcx, vel_c = torch.from_numpy(resize(ucx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(
                    resize(utcx.cpu(), [c_Nx, c_Nx], order=4)), torch.from_numpy(resize(vel.cpu(), [c_Nx, c_Nx], order=4))
                ucx, utcx, vel_c = ucx.unsqueeze(dim=0).to(device), utcx.unsqueeze(dim=0).to(device), vel_c.unsqueeze(dim=0).to(device)
                ucx, utcx = velocity_verlet_tensor(
                    ucx, utcx,
                    vel_c, dX, dT, delta_t_star, number=1, boundary_c=boundary_c
                )
                ucx, utcx = ucx.squeeze(), utcx.squeeze()
                ucx, utcx = torch.from_numpy(resize(ucx.cpu(), [Nx, Nx], order=4)), torch.from_numpy(
                    resize(utcx.cpu(), [Nx, Nx], order=4))
                res = wave_util.WaveEnergyField_tensor(ucx, utcx, vel.cpu(), dx) * dx * dx
                c_solver_acc_list.append(loss_f(res,f_solver_res[j-1]))



            # restriction
            input_restr = input[0, :3, :, :].unsqueeze(dim=0)
            for j in range(1, n_snaps):
                input_restr = torch.concat([input_restr.cpu(), vel.unsqueeze(dim=0).unsqueeze(0).cpu()], dim=1)
                input_restr = input_restr.to(device)
                vel = vel.to(device)
                output = restr_model(input_restr)  # b x 3 x w x h
                u_x, u_y, u_t_c = output[:, 0, :, :], output[:, 1, :, :], output[:, 2, :, :]
                sumv = torch.sum(torch.sum(u_x))
                u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, dt, sumv)
                res = wave_util.WaveEnergyField_tensor(u.squeeze().cpu(), ut.squeeze().cpu(), vel.cpu(), dx) * dx * dx
                model_acc_list.append(loss_f(res, f_solver_res[j-1]))
                input_restr = output.clone()

            break

    x = [i for i in range(1,n_snaps)]
    plt.plot(x, c_solver_acc_list, label="coarse solver")
    plt.plot(x, model_acc_list, label="end to end")
    plt.xlabel('iteration')
    plt.ylabel('MSE loss')
    plt.legend()
    #plt.show()
    plt.savefig("../results/run_2/good_one/acc_plot_compare.png")



if __name__ == '__main__':
    compare_end_to_end_gpu()


