import numpy as np
from scipy import fftpack


def WaveEnergyField(u,ut,c,dx):
    # Compute wave energy field
    #first line eq. 30 appendix, #Fig 2
    #pseudo-inverse of Λ -> Λ^dagger

    ux, uy = np.gradient(u,dx)
    absux = np.abs(ux)
    absuy = np.abs(uy)
    absutc = np.divide(np.abs(ut),c)
    w = np.multiply(absux,absux) + np.multiply(absuy,absuy) + np.multiply(absutc,absutc)

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