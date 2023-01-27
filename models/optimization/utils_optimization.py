import numpy as np
import torch.nn.functional as F
import torch
from generate_data.wave_propagation import pseudo_spectral, velocity_verlet_tensor
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor, \
    WaveEnergyField_tensor, WaveSol_from_EnergyComponent, WaveEnergyField
from torchmetrics.functional import mean_squared_error as MSE
from torchmetrics.functional import mean_absolute_error as MAE


def relative_frobenius_norm(a, b):
    diff = a - b
    norm_diff = torch.sqrt(torch.sum(diff**2))
    norm_b = torch.sqrt(torch.sum(b**2))
    return norm_diff / norm_b


def compute_loss(prediction,target, vel, mode = "MSE/frob(fine)"):

    prediction, target = get_wavefield(prediction, vel), get_wavefield(target,vel)

    if mode=="relative_frobenius":
        return relative_frobenius_norm(prediction, target).item()
    elif mode == "MSE":
        return MSE(prediction, target).item()
    elif mode == "MAE":
        return MAE(prediction, target).item()
    elif mode == "MSE/frob(fine)":
        return torch.linalg.norm((prediction - target) / target).item()
        # return MSE(prediction, target).item() / torch.linalg.norm(target).item()
    elif mode == "MAE/frob(fine)":
        return MAE(prediction, target).item() / torch.linalg.norm(target).item()
    else:
        raise NotImplementedError("This mode has not been implemented yet.")


def round_loss(number):
    return number #str(round(number*(10**7),5))+"e-7"


def get_solver_solution(u_n_k, n_snapshots, vel, solver="coarse"):
    # u_0_k -> b x c x w x h
    # vel -> b x w x h

    if solver == "coarse":
        small_res_scale = 2
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w, h])

        for s in range(n_snapshots):

            sol[s] = u_n_k

            a = F.upsample(u_n_k[:,0].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b = F.upsample(u_n_k[:,1].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale), mode='bilinear')
            d = F.upsample(vel, size=(w//small_res_scale, w//small_res_scale), mode='bilinear')

            u_n_k = torch.concat([a,b,b2,d],dim=1)

            u_n_k = one_iteration_velocity_verlet(u_n_k,f_delta_x=2./64., f_delta_t=1./600., delta_t_star = .06)

            a2 = F.upsample(u_n_k[:, 0].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 1].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b22 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w, w), mode='bilinear')

            u_n_k = torch.concat([a2,b2,b22], dim=1)

    elif solver == "fine":
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w//2, h//2])

        for s in range(n_snapshots):
            sol[s] = smaller_crop(u_n_k)
            u_n_k = torch.concat([u_n_k,vel], dim=1)
            u_n_k = one_iteration_pseudo_spectral(u_n_k)

    else:
        raise NotImplementedError("This solver has not been implemented yet.")

    return sol


def smaller_crop(matrix):
    # matrix -> b? x c x w x h
    if matrix.shape[-1] == 256:
        v = 64
    else:
        v = 32

    if len(matrix.shape) == 4:
        return matrix[:,:,v:-v, v:-v]
    elif len(matrix.shape) == 3:
        return matrix[:, v:-v, v:-v]
    else:
        return matrix[v:-v, v:-v]


def one_iteration_pseudo_spectral(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 20., delta_t_star = .06):

    # u_n_k -> b x c x w x h

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


def one_iteration_velocity_verlet(u_n_k,f_delta_x=2./128., f_delta_t=(2./128.)/20.,delta_t_star=.06, new_res = 128, model=None):
    # u_n_k -> b x c x w x h

    if model is None:

        vel = u_n_k[:, 3, :, :].clone()
        old_res = vel.shape[-1]
        u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0, :, :].clone(),
                                                     u_n_k[:, 1, :, :].clone(),
                                                     u_n_k[:, 2, :, :].clone(),
                                                     u_n_k[:, 3, :, :].clone(),
                                                     f_delta_x,
                                                     torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))))

        if old_res == 256:
            u = F.upsample(u.unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear').squeeze().unsqueeze(dim=0)
            u_t = F.upsample(u_t.unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear').squeeze().unsqueeze(dim=0)
            vel_crop = F.upsample(vel.unsqueeze(dim=0), size=(new_res, new_res), mode='bilinear').squeeze().unsqueeze(dim=0)

            u_prop, u_t_prop = velocity_verlet_tensor(u, u_t, vel_crop, f_delta_x, f_delta_t, delta_t_star,number=1,boundary_c="absorbing")

            u_prop = F.upsample(u_prop.unsqueeze(dim=0), size=(old_res, old_res), mode='bilinear').squeeze().unsqueeze(dim=0)
            u_t_prop = F.upsample(u_t_prop.unsqueeze(dim=0), size=(old_res, old_res), mode='bilinear').squeeze().unsqueeze(dim=0)

        else:
            u_prop, u_t_prop = velocity_verlet_tensor(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star, number=1,
                                                      boundary_c="absorbing")

        u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                          u_t_prop,
                                                          vel, f_delta_x)
        return torch.stack([u_x, u_y, u_t_c], dim=1)

    else:
        return model(u_n_k)


def get_wavefield(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t,
                                                 torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                  f_delta_x) * f_delta_x * f_delta_x

def get_wavefield_numpy(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :]
    u, u_t = WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t,
                                                 np.sum(np.sum(np.sum(u_x))))
    return WaveEnergyField(u, u_t, vel,
                                  f_delta_x) * f_delta_x * f_delta_x

