import scipy
import torch
from models import model_end_to_end
from generate_data.wave_propagation import pseudo_spectral, velocity_verlet, velocity_verlet_tensor
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import matplotlib.pyplot as plt
from models.model_utils import fetch_data_end_to_end, get_params

def parareal_scheme(model, u_n, n_parareal = 4, n_snapshots = 11):

    vel = u_n[:,3].clone().unsqueeze(dim=1)  # 1 x 1 x 128 x 128
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 128 x 128
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, width, height])

    # initial condition
    n = 0
    for k in range(n_parareal+1):
        parareal_tensor[k,n] = u_n[:,:3].clone()

    # initial guess, first iteration without parareal
    k = 0
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 128 x 128
        parareal_tensor[k,n+1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):

        parareal_terms = get_parareal_terms(model, parareal_tensor[k-1], n_snapshots, vel) # n_snapshots x b x c x w x h

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((parareal_tensor[k,n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + parareal_terms[n]
            parareal_tensor[k, n+1] = u_n1_k1

    return parareal_tensor  # k x s x b x c x w x h


def get_parareal_terms(model, parareal_tensor_k, n_snapshots, vel):
    # this can be done later computed in parallel

    parareal_terms = torch.zeros(parareal_tensor_k.shape) # n_snapshots x batch_size x channel-1 x width x height

    for s in range(n_snapshots):
        parareal_terms[s] = compute_parareal_term(model, torch.cat([parareal_tensor_k[s], vel], dim=1))

    return parareal_terms


def compute_parareal_term(model, u_n_k):

    # u_n_k -> b x c x w x h

    res_fine_solver = one_iteration_pseudo_spectral(u_n_k) #one_iteration_velocity_verlet(u_n_k)

    res_model = model(u_n_k)  # procrustes_optimization(model(u_n_k), res_fine_solver)

    return res_fine_solver - res_model


def procrustes_optimization(matrix, target):
    # matrix -> n_snapshots x 1 x 4 x 128 x 128

    procrustes_res = torch.zeros(matrix.shape)

    # channel-wise procrustes
    for c in range(matrix.shape[1]):
        m, t = matrix[0,c], target[0,c]
        omega, _ = scipy.linalg.orthogonal_procrustes(m, t)
        procrustes_res[0,c,:,:] = torch.from_numpy(omega) * m

    return procrustes_res



def one_iteration_pseudo_spectral(u_n_k):

    # TODO: change to absorbing boundary condition; this means we need to always have an area around current wavefield

    # u_n_k -> b x c x w x h

    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    delta_t_star = .06

    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0, :, :].clone(),
                                                 u_n_k[:, 1, :, :].clone(),
                                                 u_n_k[:, 2, :, :].clone(),
                                                 u_n_k[:, 3, :, :].clone(),
                                                 f_delta_x,
                                                 torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))))
    u, u_t, vel = u.squeeze().numpy(), u_t.squeeze().numpy(), u_n_k[:, 3, :, :].clone().squeeze().numpy()
    u_prop, u_t_prop = pseudo_spectral(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(torch.from_numpy(u_prop).unsqueeze(dim=0),
                                                      torch.from_numpy(u_t_prop).unsqueeze(dim=0),
                                                      torch.from_numpy(vel).unsqueeze(dim=0), f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)

def one_iteration_velocity_verlet(u_n_k):

    # u_n_k -> b x c x w x h

    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    delta_t_star = .06

    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0, :, :].clone(),
                                                 u_n_k[:, 1, :, :].clone(),
                                                 u_n_k[:, 2, :, :].clone(),
                                                 u_n_k[:, 3, :, :].clone(),
                                                 f_delta_x,
                                                 torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))))
    vel = u_n_k[:, 3, :, :].clone()
    u_prop, u_t_prop = velocity_verlet_tensor(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star,number=1,boundary_c="absorbing")
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                      u_t_prop,
                                                      vel, f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)

import numpy as np

def vis_parareal():

    # data
    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    param_dict = get_params("0")
    path = ['../data/end_to_end_bp_m_10_2000.npz']
    loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=True, batch_size=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_end_to_end.Restriction_nn(param_dict=param_dict).double().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10_2.pt'))
    model.eval()

    fig = plt.figure(figsize=(35, 8))

    with torch.no_grad():
        for i, data in enumerate(loader):

            inpt = data[0].squeeze()  # 12 x 4 x w x h
            np.savetxt("foo.csv", inpt[0,0], delimiter=",")
            vel = inpt[0,3].unsqueeze(dim=0)
            u_n = inpt[0, :, :, :].unsqueeze(dim=0)

            # coarse solver solution
            coarse_solver_tensor = get_coarse_solver_solution(u_n[:,:3,:,:],11,u_n[:,3,:,:].unsqueeze(dim=0))  # s x b x c x w x h
            for s in range(inpt.shape[0]-1):
                ax = fig.add_subplot(7, 11, 1 + s)
                wave_field = get_wavefield(coarse_solver_tensor[s], vel)
                pos = ax.imshow(wave_field)
                if s!=0: plt.colorbar(pos)
                ax.set_title("coarse solver it " + str(s),fontdict={'fontsize': 7})
                plt.axis('off')

            # parareal scheme
            parareal_tensor = parareal_scheme(model, u_n)  # k x s x b x c x w x h
            for k in range(parareal_tensor.shape[0]):
                for s in range(parareal_tensor.shape[1]):
                    wave_field = get_wavefield(parareal_tensor[k,s,:,:,:,:], vel)
                    ax = fig.add_subplot(7, 11, 11 + 11*k + s + 1)
                    pos = ax.imshow(wave_field)
                    if s != 0:
                        plt.colorbar(pos)
                        ax.set_title(torch.linalg.norm(wave_field - get_wavefield(inpt[s,:3].unsqueeze(dim=0), vel),ord='fro').item(), fontdict={'fontsize': 7}) #get frobenius norm fine vs parareal
                    plt.axis('off')

            # fine solver solution
            for s in range(inpt.shape[0]-1):
                ax = fig.add_subplot(7, 11, 67 + s)
                wave_field = get_wavefield(inpt[s,:3].unsqueeze(dim=0), vel)
                pos = ax.imshow(wave_field)
                if s != 0: plt.colorbar(pos)
                ax.set_title("fine solver it " + str(s),fontdict={'fontsize': 7})
                plt.axis('off')


            plt.show()
            break



def get_wavefield(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                           f_delta_x) * f_delta_x * f_delta_x


def get_coarse_solver_solution(u_n_k, n_snapshots, vel):
    # u_0_k -> b x c x w x h

    b, c, w, h = u_n_k.shape
    fine_sol = torch.zeros([n_snapshots, b, c, w, h])

    for s in range(n_snapshots):

        fine_sol[s] = u_n_k
        u_n_k = one_iteration_velocity_verlet(torch.cat([u_n_k, vel], dim=1))

    return fine_sol



if __name__ == '__main__':
    vis_parareal()





