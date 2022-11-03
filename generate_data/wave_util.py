import numpy as np
from scipy import fftpack
import torch

def WaveEnergyField(u,ut,c,dx):
    # Compute wave energy field

    ux, uy = np.gradient(u,dx)
    absux = np.abs(ux)
    absuy = np.abs(uy)
    absutc = np.divide(np.abs(ut),c)
    w = np.multiply(absux,absux) + np.multiply(absuy,absuy) + np.multiply(absutc,absutc)

    return w

def WaveEnergyField_tensor(u,ut,c,dx):
    # Compute wave energy field

    ux, uy = torch.gradient(u,spacing=dx)
    absux = torch.abs(ux)
    absuy = torch.abs(uy)
    absutc = torch.divide(np.abs(ut),c)
    w = torch.multiply(absux,absux) + torch.multiply(absuy,absuy) + torch.multiply(absutc,absutc)

    return w

def WaveEnergyComponentField(uS,utS,c,dx):
    # Compute wave energy component field
    wx = np.zeros(uS.shape)
    wy = np.zeros(uS.shape)
    wtc = np.zeros(uS.shape)
    for i in range(uS.shape[2]):
        wx[:,:,i],wy[:,:,i] = np.gradient(uS[:,:,i],dx)
        wtc[:,:,i] = np.divide(utS[:,:,i],c)
    return wx,wy,wtc

def WaveEnergyComponentField_end_to_end(uS,utS,c,dx):
    # Compute wave energy component field for end_to_end approach

    wx,wy = np.gradient(uS,dx)
    wtc = np.divide(utS,c)

    return wx,wy,wtc

def WaveEnergyComponentField_tensor(uS, utS, c, dx):
    # Compute wave energy component field
    wx = torch.zeros(uS.shape)
    wy = torch.zeros(uS.shape)
    wtc = torch.zeros(uS.shape)
    for b in range(uS.shape[0]):
        wx[b, :, :], wy[b, :, :] = torch.gradient(uS[b, :, :], spacing= dx)
        wtc[b, :, :] = torch.divide(utS[b, :, :], c[b,:,:])

    return wx, wy, wtc

def WaveSol_from_EnergyComponent(wx,wy,wtc,c,dx,sumv):
    # Compute wave solution components from energy component

    u = grad2func(wx,wy,dx,sumv)
    ut = np.multiply(wtc,c)
    
    return u,ut

def grad2func(vx,vy,dx,sumv):
    # Mapping gradient to functional value
    
    hatx = fftpack.fft2(vx)
    haty = fftpack.fft2(vy)

    ny,nx = vx.shape
    
    xii = 2*np.pi/(dx*nx)*fftpack.fftshift(np.linspace(-round(nx/2),round(nx/2-1),nx))
    yii = 2*np.pi/(dx*ny)*fftpack.fftshift(np.linspace(-round(ny/2),round(ny/2-1),ny))

    yiyi,xixi = np.meshgrid(xii,yii)
    radsq = np.multiply(xixi,xixi)+np.multiply(yiyi,yiyi)
    radsq[0,0] = 1
    hatv = -1j*np.divide((np.multiply(hatx,xixi)+np.multiply(haty,yiyi)),radsq)
    hatv[0,0] = sumv

    return np.real(fftpack.ifft2(hatv))

def gradient2Per(v,dx,dy):
    # Approximate gradient in periodic domain

    vx,vy = np.gradient(v,dx,dy)
    return vx, vy


def WaveSol_from_EnergyComponent_tensor(wx, wy, wtc, c, dx, sumv):
    # Compute wave solution components from energy component

    u = torch.zeros((wx.shape[0],128,128))
    for b in range(wx.shape[0]):
        u[b,:,:] = grad2func_tensor(wx[b,:,:], wy[b,:,:], dx, sumv)

    ut = torch.multiply(wtc, c)

    return u, ut


def grad2func_tensor(vx, vy, dx, sumv):
    # Mapping gradient to functional value

    hatx = torch.fft.fft2(vx)
    haty = torch.fft.fft2(vy)

    ny, nx = vx.shape

    xii = 2 * torch.pi / (dx * nx) * torch.fft.fftshift(torch.linspace(-round(nx / 2), round(nx / 2 - 1), nx))
    yii = 2 * torch.pi / (dx * ny) * torch.fft.fftshift(torch.linspace(-round(ny / 2), round(ny / 2 - 1), ny))

    yiyi, xixi = torch.meshgrid(xii, yii)
    radsq = torch.multiply(xixi, xixi) + torch.multiply(yiyi, yiyi)
    radsq[0, 0] = 1
    hatv = -1j * torch.divide((torch.multiply(hatx, xixi) + torch.multiply(haty, yiyi)), radsq)
    hatv[0, 0] = sumv

    return torch.real(torch.fft.ifft2(hatv))