import sys
sys.path.append("..")
sys.path.append("../..")
import scipy
import torch
from models.model_utils import sample_label_normal_dist
from parareal.generate_data.wave_util import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_solution(model, loss_f, data, label_distr_shift):
    # inpt -> # b x n_snaps x 4 x 256 x 256

    n_snaps = data.shape[1]  # 10
    loss_list = []

    for input_idx in random.choice(range(n_snaps - 2)):

        u_0 = data[:, input_idx, :, :, :]  # b x 4 x w x h

        # randomly sample label idx from normal distribution
        label_range = sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, -1)

        if label_range - input_idx == 1:
            label = data[:, input_idx + 1, :3, :, :].unsqueeze(dim=0).to(device)  # b x n_snaps-1 x 3 x w x h
            it = 2
        else:
            label = data[:, input_idx + 1 : label_range + 1, :3, :, :].to(device)  # b x n_snaps-1 x 3 x w x h
            it = label_range - input_idx + 1
        loss_list += parareal_scheme(model, u_0, label, loss_f, 2, it)

    return loss_list


def parareal_scheme(model, u_0, label, loss_f, n_parareal, n_snapshots):

    # u_n -> b x c x w x h
    loss_list = []
    u_n = u_0.clone()
    vel = u_n[:,3].clone().unsqueeze(dim=1)  # 1 x 1 x 500 x 500
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 500 x 500
    big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

    # initial guess, first iteration without parareal
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 512 x 512
        loss_list.append(loss_f(smaller_crop(u_n1), smaller_crop(label[:,n])))
        big_tensor[n+1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print(k)

        big_tensor[0] = u_0[:, :3].clone()
        parareal_terms = get_parareal_terms(model.to(device), big_tensor.to(device).clone().detach(), n_snapshots, vel.to(device).clone().detach()) # n_snapshots x b x c x w x h
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
        new_big_tensor[0] = u_0[:, :3].clone()

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((new_big_tensor[n].to(device), vel.to(device)), dim=1).to(device)
            u_n1_k1 = model(u_n_k1) + parareal_terms[n].to(device)
            loss_list.append(loss_f(smaller_crop(u_n1_k1), smaller_crop(label[:, n].to(device))))
            new_big_tensor[n+1] = u_n1_k1

        big_tensor = new_big_tensor.clone().detach()

    return loss_list  # k x s x b x c x w x h


def get_parareal_terms(model, big_pseudo_tensor, n_snapshots, vel):
    # this can be done later computed in parallel

    model.eval()
    with torch.no_grad():
        parareal_terms = torch.zeros(big_pseudo_tensor.shape) # n_snapshots x b x c x w x h
        for s in range(n_snapshots):
            parareal_terms[s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[s], vel], dim=1))
    model.train()
    return parareal_terms


def compute_parareal_term(model, u_n_k):

    # u_n_k -> b x c x w x h

    res_fine_solver = one_iteration_pseudo_spectral(u_n_k) #one_iteration_velocity_verlet(u_n_k)
    res_model = model(u_n_k)  # procrustes_optimization(model(u_n_k), res_fine_solver)

    return res_fine_solver.to(device) - res_model.to(device)


def procrustes_optimization(matrix, target):
    # matrix -> n_snapshots x 1 x 4 x 128 x 128

    procrustes_res = torch.zeros(matrix.shape)

    # channel-wise procrustes
    for c in range(matrix.shape[1]):
        m, t = matrix[0,c], target[0,c]
        omega, _ = scipy.linalg.orthogonal_procrustes(m, t)
        procrustes_res[0,c,:,:] = torch.from_numpy(omega) * m

    return procrustes_res



def smaller_crop(matrix):
    # matrix -> b? x c x w x h
    v = 64
    if len(matrix.shape) == 3:
        return matrix[:, v:-v, v:-v]
    elif len(matrix.shape) == 4:
        return matrix[:,:,v:-v, v:-v]
    elif len(matrix.shape) == 5:
        return matrix[:,:,:,v:-v, v:-v]
    else:
        raise NotImplementedError("This dimensionality has not been implemented yet.")


def one_iteration_pseudo_spectral(u_n_k):

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
    u_prop, u_t_prop = pseudo_spectral_tensor(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                      u_t_prop,
                                                      vel, f_delta_x)

    return torch.stack([u_x, u_y, u_t_c], dim=1)


def pseudo_spectral_tensor(u0, ut0, vel, dx, dt, Tf):
    """
    propagate wavefield using RK4 in time and spectral approx.
    of Laplacian in space
    """
    # u0 -> b x w x h
    Nt = round(abs(Tf / dt))
    c2 = torch.multiply(vel, vel)

    u = u0.to(device)
    ut = ut0.to(device)

    for i in range(Nt):
        # RK4 scheme
        k1u = ut
        k1ut = torch.multiply(c2.to(device), spectral_del_tensor(u.to(device), dx))

        k2u = ut + dt / 2 * k1ut
        k2ut = torch.multiply(c2.to(device), spectral_del_tensor(u.to(device) + dt / 2 * k1u, dx))

        k3u = ut + dt / 2 * k2ut
        k3ut = torch.multiply(c2.to(device), spectral_del_tensor(u.to(device) + dt / 2 * k2u.to(device), dx))

        k4u = ut + dt * k3ut
        k4ut = torch.multiply(c2.to(device), spectral_del_tensor(u.to(device) + dt * k3u.to(device), dx))

        u = u.to(device) + 1. / 6 * dt * (k1u.to(device) + 2 * k2u.to(device) + 2 * k3u.to(device) + k4u.to(device))
        ut = ut.to(device) + 1. / 6 * dt * (k1ut.to(device) + 2 * k2ut.to(device) + 2 * k3ut.to(device) + k4ut.to(device))

    return torch.real(u), torch.real(ut)


def spectral_del_tensor(v, dx):
    """
    evaluate the discrete Laplacian using spectral method
    """

    N1 = v.shape[-2]
    N2 = v.shape[-1]

    kx = 2 * torch.pi / (dx * N1) * torch.fft.fftshift(torch.linspace(-round(N1 / 2), round(N1 / 2 - 1), N1))
    ky = 2 * torch.pi / (dx * N2) * torch.fft.fftshift(torch.linspace(-round(N2 / 2), round(N2 / 2 - 1), N2))
    [kxx, kyy] = torch.meshgrid(kx, ky, indexing='xy')

    U = -(kxx.to(device) ** 2 + kyy.to(device) ** 2) * torch.fft.fft2(v.to(device))

    return torch.fft.ifft2(U)