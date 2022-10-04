import numpy as np
from scipy import fftpack

# Compute wave energy field
def WaveEnergyField(u,ut,c,dx):
    ux,uy = np.gradient(u,dx)
    
    absux = np.abs(ux)
    absuy = np.abs(uy)
    absutc = np.divide(np.abs(ut),c)
    print(1/0)
    w = np.multiply(absux,absux) + np.multiply(absuy,absuy) + np.multiply(absutc,absutc)
    
    return w


# Compute wave energy component field
def WaveEnergyComponentField(uS,utS,c,dx):
    wx = np.zeros(uS.shape)
    wy = np.zeros(uS.shape)
    wtc = np.zeros(uS.shape) 
    for i in range(uS.shape[2]):
        wx[:,:,i],wy[:,:,i] = np.gradient(uS[:,:,i],dx)
        wtc[:,:,i] = np.divide(utS[:,:,i],c)
        
    return wx,wy,wtc

# Compute wave solution components from energy component
def WaveSol_from_EnergyComponent(wx,wy,wtc,c,dx,sumv):

    u = grad2func(wx,wy,dx,sumv)
    ut = np.multiply(wtc,c)
    
    return u,ut


# Mapping gradient to functional value
def grad2func(vx,vy,dx,sumv):
    
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

# Approximate gradient in periodic domain
def gradient2Per(v,dx,dy):
    vx,vy = np.gradient(v,dx,dy)
    return vx,vy