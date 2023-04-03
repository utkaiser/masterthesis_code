import logging
import torch
from generate_data.change_wave_arguments import WaveSol_from_EnergyComponent_tensor, \
    WaveEnergyField_tensor, WaveSol_from_EnergyComponent, WaveEnergyField


def crop_center(img, crop_size, boundary_condition="absorbing", scaler=2):
    # crop center of img given size of crop, and scale factor
    if boundary_condition == "absorbing":
        y, x = img.shape
        startx = x // scaler - (crop_size // scaler)
        starty = y // scaler - (crop_size // scaler)

        return img[starty:starty + crop_size, startx:startx + crop_size]
    else:
        return img


def start_logger_datagen_end_to_end(output_path):
    #logger setup
    logging.basicConfig(filename=output_path + ".log",
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info("start logging")


def get_resolution_padding(boundary_condition, resolution, optimization):
    if boundary_condition == "periodic":
        res_padded = resolution
    else:  # boundary_condition == "absorbing"
        if optimization == "none":
            res_padded = resolution * 2
        else:  # optimization == "parareal"
            res_padded = resolution * 3
    return res_padded


def get_wavefield(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0], tensor[:, 1], tensor[:, 2]
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


def smaller_crop(matrix):
    # matrix -> b? x c x w x h
    if matrix.shape[-1] == 256:
        v = 64
    else:
        v = 32

    if len(matrix.shape) == 3:
        return matrix[:, v:-v, v:-v]
    elif len(matrix.shape) == 4:
        return matrix[:,:,v:-v, v:-v]
    elif len(matrix.shape) == 5:
        return matrix[:,:,:,v:-v, v:-v]
    elif len(matrix.shape) == 6:
        return matrix[:,:,:,:,v:-v, v:-v]

    else:
        return matrix[v:-v, v:-v]