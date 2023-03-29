import scipy
import torch
from scipy.io import savemat
import numpy as np
import glob
import os
from generate_data.initial_conditions import get_velocity_crop
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor, velocity_verlet_tensor, pseudo_spectral
from models.model_utils import smaller_crop, get_wavefield_numpy, get_wavefield
import torch.nn.functional as F


def change_npy_to_mat(dir_name = ""):
    # changes all files in folder from npy format to mat format

    a = np.load("vis_files/vel.npy")
    scipy.io.savemat('vel.mat', dict(res=a))
    # d = np.load("diagonal_fine.npy")
    # vel = get_velocity_crop(128, 1, "marmousi", "absorbing", "none")[0]


if __name__ == '__main__':
    change_npy_to_mat()

def get_ticks_fine(tensor, vel):
    # tensor -> 1 x s x c x w x h
    # vel -> w x h

    ticks = []
    for s in range(tensor.shape[1]):
        img = get_wavefield(tensor[:,s],vel)
        ticks.append([img.min().item(), (img.max().item() + img.min().item()) / 2 ,img.max().item()])
    return ticks  # s x 3


def get_solver_solution(u_n_k, n_snapshots, vel, solver="coarse"):
    # u_0_k -> b x c x w x h
    # vel -> b x w x h

    if solver == "coarse":
        small_res_scale = 2
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([b, n_snapshots, c, w, h])
        for s in range(n_snapshots):

            sol[0,s] = u_n_k.squeeze()

            a = F.upsample(u_n_k[:,0].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b1 = F.upsample(u_n_k[:,1].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale), mode='bilinear')
            d = F.upsample(vel, size=(w//small_res_scale, w//small_res_scale), mode='bilinear')

            u_n_k = torch.concat([a,b1,b2,d],dim=1)

            u_n_k = one_iteration_velocity_verlet(u_n_k,f_delta_x=2./64., f_delta_t=1./600., delta_t_star = .06)

            a2 = F.upsample(u_n_k[:, 0].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 1].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b22 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w, w), mode='bilinear')

            u_n_k = torch.concat([a2,b2,b22], dim=1)

    elif solver == "fine":
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w//2, h//2])

        for s in range(n_snapshots):
            sol[s,:] = smaller_crop(u_n_k)
            u_n_k = torch.concat([u_n_k,vel], dim=1)
            u_n_k = one_iteration_pseudo_spectral(u_n_k)

    else:
        raise NotImplementedError("This solver has not been implemented yet.")

    return sol


def one_iteration_pseudo_spectral(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 20., delta_t_star = .06):

    # u_n_k -> b x c x w x h

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
                                                      vel.unsqueeze(dim=0), f_delta_x)
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