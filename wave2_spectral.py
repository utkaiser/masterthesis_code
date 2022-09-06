import numpy as np
from numpy import fft

def spectral_del(v,dx):
    """
    evaluate the discrete Laplacian using spectral method
    """

    N1 = v.shape[0]
    N2 = v.shape[1]
    
    kx = 2*np.pi/(dx*N1)*fft.fftshift(np.linspace(-round(N1/2),round(N1/2-1),N1))
    ky = 2*np.pi/(dx*N2)*fft.fftshift(np.linspace(-round(N2/2),round(N2/2-1),N2))
    [kxx,kyy] = np.meshgrid(kx,ky)

    U = -(kxx**2+kyy**2) * fft.fft2(v)

    return fft.ifft2(U)

def wave2(u0,ut0,vel,dx,dt,Tf):
    """
    propagate wavefield using velocity Verlet in time and spectral approx.
    of Laplacian in space
    """

    Nt = round(abs(Tf/dt))
    c2 = np.multiply(vel,vel)
    
    u = u0
    ut = ut0
    
    for i in range(Nt):
        # Velocity Verlet
        ddxou = spectral_del(u,dx)
        u = u + dt*ut + 0.5*dt**2*np.multiply(c2,ddxou)
        ddxu = spectral_del(u,dx)
        ut = ut + 0.5*dt*np.multiply(c2,ddxou+ddxu)
    
    return np.real(u), np.real(ut)


def wave2s(u0,ut0,vel,dx,dt,Tf):
    """
    propagate wavefield using RK4 in time and spectral approx.
    of Laplacian in space
    """

    Nt = round(abs(Tf/dt))
    c2 = np.multiply(vel,vel)
    
    u = u0
    ut = ut0
    
    for i in range(Nt):
        # RK4 scheme
        k1u = ut
        k1ut = np.multiply(c2,spectral_del(u,dx))
        
        k2u = ut + dt/2*k1ut
        k2ut = np.multiply(c2,spectral_del(u+dt/2*k1u,dx))
        
        k3u = ut + dt/2*k2ut
        k3ut = np.multiply(c2,spectral_del(u+dt/2*k2u,dx))
        
        k4u = ut + dt*k3ut
        k4ut = np.multiply(c2,spectral_del(u+dt*k3u,dx))
        
        u = u + 1./6*dt*(k1u + 2*k2u + 2*k3u + k4u)
        ut = ut + 1./6*dt*(k1ut + 2*k2ut + 2*k3ut + k4ut)
    
    return np.real(u), np.real(ut)

