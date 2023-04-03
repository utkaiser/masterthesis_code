import torch
from generate_data.change_wave_arguments import WaveSol_from_EnergyComponent_tensor,WaveEnergyComponentField_tensor
from generate_data.utils import smaller_crop
from generate_data.wave_propagation import pseudo_spectral_tensor, velocity_verlet_tensor
import torch.nn.functional as F


def one_iteration_pseudo_spectral_tensor(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 20., delta_t_star = .06):

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