import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import random
import torch
from generate_data.change_wave_arguments import WaveEnergyComponentField_tensor


def initial_condition_gaussian(
        vel,
        resolution,
        boundary_condition,
        optimization,
        mode,
        res_padded
):
    '''
    Parameters
    ----------
    vel : (numpy tensor) velocity profile
    resolution : (int) resolution of actual area to propagate wave
    boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"
    optimization : (string) optimization technique; "parareal" or "none"
    mode : (string) defines initial condition representation; "physical_components" or "energy_components"
    res_padded : (int) resolution of padded area to propagate wave, > resolution in case of "parareal" and / or "absorbing"

    Returns
    -------
    generates a Gaussian pulse to be used as an initial condition for our end-to-end model to advance waves
    '''

    dx, width, center_x, center_y = _get_init_cond_settings(resolution, boundary_condition, optimization)
    u0, ut0 = init_pulse_gaussian(width, res_padded, center_x, center_y)

    if mode == "physical_components":
        return u0, ut0
    else:  # energy_components
        u0, ut0 = torch.from_numpy(u0).unsqueeze(dim=0), torch.from_numpy(ut0).unsqueeze(dim=0)
        wx, wy, wtc = WaveEnergyComponentField_tensor(u0, ut0, vel.unsqueeze(dim=0), dx=dx)
        return torch.stack([wx, wy, wtc], dim=1)


def _get_init_cond_settings(
        resolution,
        boundary_condition,
        optimization
):
    '''
    Parameters
    ----------
    resolution : (int) resolution of actual area to propagate wave
    boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    get settings of initial condition depending on boundary_condition and optimization technique;
    settings include random width and centers of initial conditon
    '''

    factor_width = random.random() * 2 - 1
    factor_center_x = random.random() * 2 - 1
    factor_center_y = random.random() * 2 - 1

    if optimization == "parareal": factor_start_point_wave = 2
    else: factor_start_point_wave = 1

    if boundary_condition == "periodic":
        factor = 2 if optimization == "parareal" else 1
        center_x, center_y = factor_center_x * .45 / factor_start_point_wave / factor, factor_center_y * .45  / factor_start_point_wave / factor
        if resolution == 128:
            width = 1000 + factor_width*200
        elif resolution == 256:
            width = 7000 + factor_width*500
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    elif boundary_condition == "absorbing":
        center_x, center_y = ((factor_center_x * .45) / 2) / factor_start_point_wave, ((factor_center_y * .45) / 2) / factor_start_point_wave
        if resolution == 128:
            if optimization == "none":
                width = 5600 + factor_width * 500
            else:
                width = 20000 + factor_width * 800
        elif resolution == 256:
            width = 9000 + factor_width * 500
        else: raise NotImplementedError("Parameter for initial condition not implemented.")

    else: raise NotImplementedError("Boundary condition for initial condition not implemented.")

    return 2.0 / 128.0, width, center_x, center_y


def init_pulse_gaussian(
        width,
        res_padded,
        center_x,
        center_y
):
    '''

    Parameters
    ----------
    width : (float) width of initial pulse
    res_padded : (int) padded resolution
    center_x : (float) center of initial pulse in x_1 direction
    center_y : (float) center of initial pulse in x_2 direction

    Returns
    -------
    generates initial Gaussian pulse  (see formula in paper)
    '''

    xx, yy = np.meshgrid(np.linspace(-1, 1, res_padded), np.linspace(-1, 1, res_padded))
    u0 = np.exp(-width * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


