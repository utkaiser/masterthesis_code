import numpy as np
from scipy import fftpack
import logging


def WaveEnergyField(u, ut, c, dx):
    # Compute wave energy field

    ux, uy = np.gradient(u, dx)
    absux = np.abs(ux)
    absuy = np.abs(uy)
    absutc = np.divide(np.abs(ut), c)
    w = np.multiply(absux, absux) + np.multiply(absuy, absuy) + np.multiply(absutc, absutc)

    return w


def WaveEnergyField_tensor(u, ut, c, dx):
    # Compute wave energy field

    ux, uy = torch.gradient(u, spacing=dx)
    absux = torch.abs(ux)
    absuy = torch.abs(uy)
    absutc = torch.divide(torch.abs(ut), c)
    w = torch.multiply(absux, absux) + torch.multiply(absuy, absuy) + torch.multiply(absutc, absutc)

    return w


def WaveEnergyComponentField(uS, utS, c, dx):
    # Compute wave energy component field
    wx = np.zeros(uS.shape)
    wy = np.zeros(uS.shape)
    wtc = np.zeros(uS.shape)
    for i in range(uS.shape[2]):
        wx[:, :, i], wy[:, :, i] = np.gradient(uS[:, :, i], dx)
        wtc[:, :, i] = np.divide(utS[:, :, i], c)
    return wx, wy, wtc


def WaveEnergyComponentField_end_to_end(uS, utS, c, dx):
    # Compute wave energy component field for end_to_end approach

    wx, wy = np.gradient(uS, dx)
    wtc = np.divide(utS, c)

    return wx, wy, wtc


def WaveEnergyComponentField_tensor(uS, utS, c, dx):
    # Compute wave energy component field
    wx = torch.zeros(uS.shape)
    wy = torch.zeros(uS.shape)
    wtc = torch.zeros(uS.shape)
    for b in range(uS.shape[0]):
        wx[b, :, :], wy[b, :, :] = torch.gradient(uS[b, :, :], spacing=dx)
        wtc[b, :, :] = torch.divide(utS[b, :, :], c[b, :, :])

    return wx, wy, wtc


def WaveSol_from_EnergyComponent(wx, wy, wtc, c, dx, sumv):
    # Compute wave solution components from energy component

    u = grad2func(wx, wy, dx, sumv)
    ut = np.multiply(wtc, c)

    return u, ut


def grad2func(vx, vy, dx, sumv):
    # Mapping gradient to functional value

    hatx = fftpack.fft2(vx)
    haty = fftpack.fft2(vy)

    ny, nx = vx.shape

    xii = 2 * np.pi / (dx * nx) * fftpack.fftshift(np.linspace(-round(nx / 2), round(nx / 2 - 1), nx))
    yii = 2 * np.pi / (dx * ny) * fftpack.fftshift(np.linspace(-round(ny / 2), round(ny / 2 - 1), ny))

    yiyi, xixi = np.meshgrid(xii, yii)
    radsq = np.multiply(xixi, xixi) + np.multiply(yiyi, yiyi)
    radsq[0, 0] = 1
    hatv = -1j * np.divide((np.multiply(hatx, xixi) + np.multiply(haty, yiyi)), radsq)
    hatv[0, 0] = sumv

    return np.real(fftpack.ifft2(hatv))


def gradient2Per(v, dx, dy):
    # Approximate gradient in periodic domain

    vx, vy = np.gradient(v, dx, dy)
    return vx, vy


def WaveSol_from_EnergyComponent_tensor(wx, wy, wtc, c, dx, sumv):
    # Compute wave solution components from energy component

    u = torch.zeros((wx.shape[0], wx.shape[-1], wx.shape[-1]))

    for b in range(wx.shape[0]):
        u[b, :, :] = grad2func_tensor(wx[b, :, :], wy[b, :, :], dx, sumv)


    ut = torch.multiply(wtc, c)

    return u, ut


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad2func_tensor(vx, vy, dx, sumv):
    # Mapping gradient to functional value

    hatx = torch.fft.fft2(vx)
    haty = torch.fft.fft2(vy)

    ny, nx = vx.shape

    xii = 2 * torch.pi / (dx * nx) * torch.fft.fftshift(torch.linspace(-round(nx / 2), round(nx / 2 - 1), nx))
    yii = 2 * torch.pi / (dx * ny) * torch.fft.fftshift(torch.linspace(-round(ny / 2), round(ny / 2 - 1), ny))

    yiyi, xixi = torch.meshgrid(xii, yii,indexing="xy")

    radsq = torch.multiply(xixi, xixi) + torch.multiply(yiyi, yiyi)
    radsq[0, 0] = 1
    hatv = -1j * torch.divide(
        (torch.multiply(hatx.to(device), xixi.to(device)) + torch.multiply(haty.to(device), yiyi.to(device))),
        radsq.to(device))
    hatv[0, 0] = sumv

    return torch.real(torch.fft.ifft2(hatv))


def crop_center(img, crop_size, boundary_condition="absorbing", scaler=2):
    # crop center of img given size of crop, and scale factor
    if boundary_condition == "absorbing":
        y, x = img.shape
        startx = x // scaler - (crop_size // scaler)
        starty = y // scaler - (crop_size // scaler)

        return img[starty:starty + crop_size, startx:startx + crop_size]
    else:
        return img


def start_logger_datagen_end_to_end(index):
    #logger setup
    logging.basicConfig(filename="../results/datagen/datagen_"+index+".log",
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info("start logging")



def get_datagen_end_to_end_params(param_dict):
    return param_dict["total_time"], param_dict["delta_t_star"], param_dict["f_delta_x"], param_dict["f_delta_t"], \
           param_dict["n_snaps"], param_dict["res_scaler"]




