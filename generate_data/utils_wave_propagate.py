import torch
import torch.nn.functional as F

import sys
sys.path.append("..")
sys.path.append("../..")

from generate_data.change_wave_arguments import (
    WaveEnergyComponentField_tensor,
    WaveSol_from_EnergyComponent_tensor,
)
from generate_data.wave_propagation import (
    pseudo_spectral_tensor,
    velocity_verlet_tensor,
)


def one_iteration_pseudo_spectral_tensor(
    u_n_k, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20.0, delta_t_star=0.06
):
    """

    Parameters
    ----------
    u_n_k : (pytorch tensor) wave representation as energy components
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size
    delta_t_star : (float) time step a solver propagates a wave and solvers are compared

    Returns
    -------
    propagates a wave for one time step delta_t_star using the pseudo-spectral method
    """

    u, u_t = WaveSol_from_EnergyComponent_tensor(
        u_n_k[:, 0, :, :].clone(),
        u_n_k[:, 1, :, :].clone(),
        u_n_k[:, 2, :, :].clone(),
        u_n_k[:, 3, :, :].clone(),
        f_delta_x,
        torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))),
    )
    vel = u_n_k[:, 3, :, :].clone()
    u_prop, u_t_prop = pseudo_spectral_tensor(
        u, u_t, vel, f_delta_x, f_delta_t, delta_t_star
    )
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(
        u_prop, u_t_prop, vel.unsqueeze(dim=1), f_delta_x
    )
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def one_iteration_velocity_verlet_tensor(
    u_n_k,
    c_delta_x=2.0 / 64.0,
    c_delta_t=1.0 / 600.0,
    delta_t_star=0.06,
    new_res=128,
):
    """
    Parameters
    ----------
    u_n_k : (pytorch tensor) wave representation as energy components
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size
    delta_t_star : (float) time step a solver propagates a wave and solvers are compared
    new_res : (int) how much to downsample the input to (this is not a factor, but resolution)

    Returns
    -------
    one step of velocity verlet either using just the method, or our end-to-end model
    """

    vel = u_n_k[:, 3, :, :].clone()
    u, u_t = WaveSol_from_EnergyComponent_tensor(
        u_n_k[:, 0].clone(),
        u_n_k[:, 1].clone(),
        u_n_k[:, 2].clone(),
        u_n_k[:, 3].clone(),
        c_delta_x,
        torch.sum(torch.sum(torch.sum(u_n_k[:, 0].clone()))),
    )

    u_prop, u_t_prop = velocity_verlet_tensor(
        u,
        u_t,
        vel,
        c_delta_x,
        c_delta_t,
        delta_t_star,
        number=1,
        boundary_c="absorbing",
    )

    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(
        u_prop, u_t_prop, vel.unsqueeze(dim=0), c_delta_x
    )
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def resize_to_coarse(u_n, res_coarse):
    # u_n_k.shape: b x c x w x h

    return u_n[:, :, 32:-32, 32:-32]

def resize_to_coarse_interp(u_n, res_coarse):
    # u_n_k.shape: b x c x w x h

    return F.upsample(u_n, size=(res_coarse, res_coarse), mode='bilinear')