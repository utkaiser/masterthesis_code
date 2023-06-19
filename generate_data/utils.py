import logging
import torch
from generate_data.change_wave_arguments import WaveSol_from_EnergyComponent_tensor, \
    WaveEnergyField_tensor, WaveSol_from_EnergyComponent, WaveEnergyField
import numpy as np


def crop_center(
        img,
        crop_size,
        scaler = 2
):
    '''
    Parameters
    ----------
    img : (numpy / pytorch tensor) input image to crop
    crop_size : (int) size of crop
    boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"
    scaler : scale factor

    Returns
    -------
    crop center of img given size of crop, and scale factor
    '''

    y, x = img.shape
    startx = x // scaler - (crop_size // scaler)
    starty = y // scaler - (crop_size // scaler)

    return img[starty:starty + crop_size, startx:startx + crop_size]


def crop_center_tensor(
        img,
        crop_size,
        scaler = 2
):
    '''
    Parameters
    ----------
    img : (numpy / pytorch tensor) input image to crop
    crop_size : (int) size of crop
    boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"
    scaler : scale factor

    Returns
    -------
    crop center of img given size of crop, and scale factor
    '''

    b, c, y, x = img.shape
    startx = x // scaler - (crop_size // scaler)
    starty = y // scaler - (crop_size // scaler)

    return img[:,:,starty:starty + crop_size, startx:startx + crop_size]



def start_logger_datagen_end_to_end(
        output_path
):
    '''
    Parameters
    ----------
    output_path : logger path; file ends with ".log"

    Returns
    -------
    sets up logger environment
    '''
    logging.basicConfig(filename=output_path + ".log",
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info("start logging")


def get_resolution_padding(
        resolution,
        optimization
):
    '''
    Parameters
    ----------
    resolution : (int) resolution of actual area to propagate wave
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    padding resolution parameter as we need to add area around actual resolution area
    in case of "parareal" and / or "absorbing" when using pseudo-spectral method
    '''

    if optimization == "none":
        return resolution * 2  # for absorbing boundary conditions
    else:  # optimization == "parareal"
        return resolution * 3

def get_wavefield(
        wave_representation,
        vel,
        delta_x=2.0 / 128.0,
        delta_t=(2.0 / 128.0) / 20
):
    '''
    Parameters
    ----------
    wave_representation : (pytorch tensor) wave energy components (as batch)
    vel : (pytorch tensor) velocity profile
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size

    Returns
    -------
    transformation from energy components to energy-semi norm (tensor)
    '''

    u_x, u_y, u_t_c = wave_representation[:, 0], wave_representation[:, 1], wave_representation[:, 2]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, delta_t,
                                                 torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                  delta_x) * delta_x * delta_x


def get_wavefield_numpy(
        wave_representation,
        vel,
        f_delta_x=2.0 / 128.0,
        f_delta_t=(2.0 / 128.0) / 20
):
    '''
    Parameters
    ----------
    wave_representation : (numpy tensor) wave energy components (as batch)
    vel : (numpy tensor) velocity profile
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size

    Returns
    -------
    transformation from energy components to energy-semi norm (numpy)
    '''

    u_x, u_y, u_t_c = wave_representation[0, :, :], wave_representation[1, :, :], wave_representation[2, :, :]
    u, u_t = WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t,
                                          np.sum(np.sum(np.sum(u_x))))
    return WaveEnergyField(u, u_t, vel,
                           f_delta_x) * f_delta_x * f_delta_x


def smaller_crop(
        tensor
):
    '''
    Parameters
    ----------
    tensor : (numpy / pytorch tensor) tensor to crop; dimensionality can vary

    Returns
    -------
    center crop tensor by factor of 2
    '''

    if tensor.shape[-1] == 256:
        v = 64
    else:
        v = 32

    if len(tensor.shape) == 3:
        return tensor[:, v:-v, v:-v]
    elif len(tensor.shape) == 4:
        return tensor[:, :, v:-v, v:-v]
    elif len(tensor.shape) == 5:
        return tensor[:, :, :, v:-v, v:-v]
    elif len(tensor.shape) == 6:
        return tensor[:, :, :, :, v:-v, v:-v]

    else:
        return tensor[v:-v, v:-v]