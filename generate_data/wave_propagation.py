import numpy as np
import torch
from skimage.transform import resize # for coarsening
from scipy import fft

def parallel_compute(u, ut, vel, vel_c, f_delta_x, c_delta_x, f_delta_t, c_delta_t, delta_t_star):

    n_snapshots = u.shape[2]
    resolution_c, resolution_f = vel_c.shape[0], vel.shape[0]

    # pre-allocate arrays for output
    uf = np.zeros([resolution_f,resolution_f,n_snapshots])
    utf = np.zeros([resolution_f,resolution_f,n_snapshots])
    uc = np.zeros([resolution_c,resolution_c,n_snapshots])
    utc = np.zeros([resolution_c,resolution_c,n_snapshots])

    # parallel loop, each rhs is independent of lhs
    for j in range(n_snapshots):

        restriction_u, restriction_ut = resize(u[:, :, j], [resolution_c, resolution_c], order=4),\
                                        resize(ut[:, :, j], [resolution_c, resolution_c], order=4)

        # coarse solver propagation using finite-difference time-domain method
        uc[:,:,j], utc[:,:,j] = velocity_verlet(restriction_u, restriction_ut,
                                                vel_c, c_delta_x, c_delta_t, delta_t_star)

        # fine solver propagation using pseudo spectral method
        uf[:,:,j], utf[:,:,j] = pseudo_spectral(u[:, :, j], ut[:, :, j], vel, f_delta_x, f_delta_t, delta_t_star)

    return uc,utc,uf,utf



def velocity_verlet(u0, ut0, vel, dx, dt, delta_t_star):
    """
    Wave solution propagator
    propagate wavefield using velocity Verlet in time and the second order
    discrete Laplacian in space
    found eq. 10 in paper
    """

    Nt = round(abs(delta_t_star / dt))
    c2 = np.multiply(vel,vel)
    u = u0
    ut = ut0

    for i in range(Nt):

        # Velocity Verlet
        ddxou = periLaplacian(u,dx)
        u = u + dt*ut + 0.5*dt**2*np.multiply(c2,ddxou)
        ddxu = periLaplacian(u,dx)
        ut = ut + 0.5*dt*np.multiply(c2,ddxou+ddxu)
    
    return u, ut


def velocity_verlet_tensor(u0, ut0, vel, dx, dt, delta_t_star, number=0):
    """
    Wave solution propagator
    propagate wavefield using velocity Verlet in time and the second order
    discrete Laplacian in space
    found eq. 10 in paper
    """

    Nt = round(abs(delta_t_star / dt))
    c2 = torch.mul(vel, vel)
    u, ut = u0, ut0

    for i in range(Nt):
        # Velocity Verlet

        ddxou = periLaplacian_tensor(u, dx, number)
        u = u + dt * ut + 0.5 * dt ** 2 * torch.mul(c2, ddxou)
        ddxu = periLaplacian_tensor(u, dx, number)
        ut = ut + 0.5 * dt * torch.mul(c2, ddxou + ddxu)

    return u,ut

def periLaplacian(v,dx):
    """
    Define periodic Laplacian
    evaluate discrete Laplacian with periodic boundary condition
    """

    Lv = (np.roll(v,1,axis=1) - 2*v + np.roll(v,-1,axis=1))/(dx**2)+\
         (np.roll(v,1,axis=0) - 2*v + np.roll(v,-1,axis=0))/(dx**2)

    return Lv

def periLaplacian_tensor(v,dx, number):
    """
    Define periodic Laplacian
    evaluate discrete Laplacian with periodic boundary condition
    """

    Lv = (torch.roll(v,1,dims=1+number) - 2*v + torch.roll(v,-1,dims=1+number))/(dx**2)+\
         (torch.roll(v,1,dims=0+number) - 2*v + torch.roll(v,-1,dims=0+number))/(dx**2)

    return Lv

def pseudo_spectral(u0, ut0, vel, dx, dt, Tf):
    """
    propagate wavefield using RK4 in time and spectral approx.
    of Laplacian in space
    """

    Nt = round(abs(Tf / dt))
    c2 = np.multiply(vel, vel)

    u = u0
    ut = ut0

    for i in range(Nt):
        # RK4 scheme
        k1u = ut
        k1ut = np.multiply(c2, spectral_del(u, dx))

        k2u = ut + dt / 2 * k1ut
        k2ut = np.multiply(c2, spectral_del(u + dt / 2 * k1u, dx))

        k3u = ut + dt / 2 * k2ut
        k3ut = np.multiply(c2, spectral_del(u + dt / 2 * k2u, dx))

        k4u = ut + dt * k3ut
        k4ut = np.multiply(c2, spectral_del(u + dt * k3u, dx))

        u = u + 1. / 6 * dt * (k1u + 2 * k2u + 2 * k3u + k4u)
        ut = ut + 1. / 6 * dt * (k1ut + 2 * k2ut + 2 * k3ut + k4ut)

    return np.real(u), np.real(ut)

def spectral_del(v, dx):
    """
    evaluate the discrete Laplacian using spectral method
    """

    N1 = v.shape[0]
    N2 = v.shape[1]

    kx = 2 * np.pi / (dx * N1) * fft.fftshift(np.linspace(-round(N1 / 2), round(N1 / 2 - 1), N1))
    ky = 2 * np.pi / (dx * N2) * fft.fftshift(np.linspace(-round(N2 / 2), round(N2 / 2 - 1), N2))
    [kxx, kyy] = np.meshgrid(kx, ky)

    U = -(kxx ** 2 + kyy ** 2) * fft.fft2(v)

    return fft.ifft2(U)




