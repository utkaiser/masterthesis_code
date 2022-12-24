import numpy as np
import torch
from skimage.transform import resize # for coarsening
from scipy import fft
from generate_data.wave_util import WaveEnergyField_tensor
import matplotlib.pyplot as plt


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



def velocity_verlet_tensor(u0, ut0, vel, dx, dt, delta_t_star, number=0, boundary_c='periodic',tj=0):
    """
    Wave solution propagator
    propagate wavefield using velocity Verlet in time and the second order
    discrete Laplacian in space
    found eq. 10 in paper

    u0 shape: b x w_c x h_c
    vel shape: b x w_c x h_c
    """

    Nt = round(abs(delta_t_star / dt))
    c2 = torch.mul(vel, vel)

    u, ut = u0, ut0
    f_delta_t = 2.0 / 128.0

    if boundary_c == 'periodic':

        for i in range(Nt):
            # Velocity Verlet

            ddxou = periLaplacian_tensor(u, dx, number)
            u = u + dt * ut + 0.5 * dt ** 2 * torch.mul(c2, ddxou)
            ddxu = periLaplacian_tensor(u, dx, number)
            ut = ut + 0.5 * dt * torch.mul(c2, ddxou + ddxu)

        return u,ut

    elif boundary_c == 'absorbing':

        # shape: u, ut -> b x w_c x h_c

        Ny, Nx = u0.shape[-1] - 1, u0.shape[-2] - 1

        lambda_v = abs(dt / dx)
        lambda2 = lambda_v ** 2
        lambdaC2 = lambda2 * c2

        a = dx / (dx + abs(dt))

        # Euler step to generate u1 from u0 and ut0
        uneg1 = u0 - dt * ut0
        u2 = u0.clone()
        u1 = u0.clone()
        u0 = uneg1.clone()

        for k in range(Nt):
            # wave equation update
            u2[:,1: Ny, 1: Nx] = 2 * u1[:,1: Ny, 1: Nx] - u0[:,1: Ny, 1: Nx] + lambdaC2[:,1: Ny, 1: Nx] * \
                                  (u1[:,2:Ny + 1, 1:Nx] + u1[:,0: Ny - 1, 1: Nx] + u1[:,1: Ny, 2: Nx + 1] + u1[:,1: Ny,0: Nx - 1] - 4 * u1[:,1: Ny,1: Nx])

            # # absorbing boundary update (Engquist-Majda ABC second order)
            Ny, Nx = Ny - 1, Nx - 1
            u2[:,-1, 1:Nx + 1] = a * (
                        -u2[:,Ny, 1:Nx + 1] + 2 * u1[:,-1, 1:Nx + 1] - u0[:,-1, 1:Nx + 1] + 2 * u1[:,Ny, 1:Nx + 1] - u0[:,Ny,
                                                                                                             1:Nx + 1] +
                        lambda_v * (u2[:,Ny, 1:Nx + 1] - u0[:,Ny, 1:Nx + 1] + u0[:,-1, 1:Nx + 1]) +
                        .5 * lambda2 * (u0[:,-1, 2:Nx + 2] - 2 * u0[:,-1, 1:Nx + 1] + u0[:,-1, 0:Nx] +
                                        u2[:,Ny, 2:Nx + 2] - 2 * u2[:,Ny, 1:Nx + 1] + u2[:,Ny, 0:Nx]))

            u2[:,0, 1:Nx + 1] = a * (
                        -u2[:,1, 1:Nx + 1] + 2 * u1[:,0, 1:Nx + 1] - u0[:,0, 1:Nx + 1] + 2 * u1[:,1, 1:Nx + 1] - u0[:,1,
                                                                                                         1:Nx + 1] +
                        lambda_v * (u2[:,1, 1:Nx + 1] - u0[:,1, 1:Nx + 1] + u0[:,0, 1:Nx + 1]) +
                        .5 * lambda2 * (u0[:,0, 2:Nx + 2] - 2 * u0[:,0, 1:Nx + 1] + u0[:,0, 0:Nx] +
                                        u2[:,1, 2:Nx + 2] - 2 * u2[:,1, 1:Nx + 1] + u2[:,1, 0:Nx]))

            u2[:,1:Ny + 1, -1] = a * (
                        -u2[:,1:Ny + 1, Nx] + 2 * u1[:,1:Ny + 1, Nx] - u0[:,1:Ny + 1, Nx] + 2 * u1[:,1:Ny + 1, Nx + 1] - u0[:,
                                                                                                                 1:Ny + 1,
                                                                                                                 Nx + 1] +
                        lambda_v * (u2[:,1:Ny + 1, Nx] - u0[:,1:Ny + 1, Nx] + u0[:,1:Ny + 1, Nx + 1]) +
                        .5 * lambda2 * (u0[:,2:Ny + 2, Nx + 1] - 2 * u0[:,1:Ny + 1, Nx + 1] + u0[:,0:Ny, Nx + 1] +
                                        u2[:,2:Ny + 2, Nx] - 2 * u2[:,1:Ny + 1, Nx] + u2[:,0:Ny, Nx]))

            u2[:,1:Ny + 1, 0] = a * (
                        -u2[:,1:Ny + 1, 1] + 2 * u1[:,1:Ny + 1, 1] - u0[:,1:Ny + 1, 1] + 2 * u1[:,1:Ny + 1, 0] - u0[:,1:Ny + 1,
                                                                                                         0] +
                        lambda_v * (u2[:,1:Ny + 1, 1] - u0[:,1:Ny + 1, 1] + u0[:,1:Ny + 1, 0]) +
                        .5 * lambda2 * (u0[:,2:Ny + 2, 0] - 2 * u0[:,1:Ny + 1, 0] + u0[:,0:Ny, 0] +
                                        u2[:,2:Ny + 2, 1] - 2 * u2[:,1:Ny + 1, 1] + u2[:,0:Ny, 1]))

            # corners
            u2[:,-1, 0] = a * (u1[:,-1, 0] - u2[:,Ny, 0] + u1[:,Ny, 0] +
                             lambda_v * (u2[:,Ny, 0] - u1[:,-1, 0] + u1[:,Ny, 0]))
            u2[:,0, 0] = a * (u1[:,0, 0] - u2[:,1, 0] + u1[:,1, 0] +
                            lambda_v * (u2[:,1, 0] - u1[:,0, 0] + u1[:,1, 0]))
            u2[:,0, -1] = a * (u1[:,0, -1] - u2[:,0, Nx] + u1[:,0, Nx] +
                             lambda_v * (u2[:,0, Nx] - u1[:,0, -1] + u1[:,0, Nx]))
            u2[:,-1, -1] = a * (u1[:,-1, -1] - u2[:,Ny, -1] + u1[:,Ny, -1] +
                              lambda_v * (u2[:,Ny, -1] - u1[:,-1, -1] + u1[:,Ny, -1]))

            # update grids
            u, ut = u2.clone(), (u2 - u0) / (2 * dt)
            u0 = u1.clone()
            u1 = u2.clone()
            Ny, Nx = Ny + 1, Nx + 1

        return u, ut

    else:
        raise NotImplementedError("this boundary condition is not implemented")

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



    # path = "../old_code/ABC_code/Richard_ABC"
    #
    # for i in range(1,3):
    #     a = np.loadtxt(open(path+"/test"+str(i)+".csv", 'rb'), delimiter=',', skiprows=1)
    #     plt.imshow(a)
    #     plt.show()

# for j in range(1,Nx):
# bottom
# u2[-1, j]=a*(-u2[Ny,j] + 2 * u1[-1,j] - u0[-1, j] + 2 * u1[Ny, j] - u0[Ny, j] +
#             lambda_v * (u2[Ny,j] - u0[Ny, j] + u0[-1, j]) +
#             .5 * lambda2 * (u0[-1, j + 1] - 2 * u0[-1, j] + u0[-1, j - 1] +
#                             u2[Ny,j+1] - 2 * u2[Ny,j] + u2[Ny,j-1]))


# top
# u2[0,  j]=a*(-u2[1,j] + 2 * u1[0,j] - u0[0, j] + 2 * u1[1, j] - u0[1, j] +
#              lambda_v * (u2[1,j] - u0[1, j] + u0[0, j]) +
#              .5 * lambda2 * (u0[0, j + 1] - 2 * u0[0, j] + u0[0, j - 1] +
#                              u2[1,j+1] - 2 * u2[1,j] + u2[1,j-1]))


#for i in range(1,Ny):
            # right
            # u2[i,-1]=a*(-u2[i,Nx] + 2 * u1[i,Nx] - u0[i, Nx] + 2 * u1[i, Nx + 1] - u0[i, Nx + 1] +
            #             lambda_v * (u2[i, Nx] - u0[i, Nx] + u0[i, Nx + 1]) +
            #             .5 * lambda2 * (u0[i + 1, Nx + 1] - 2 * u0[i, Nx + 1] + u0[i - 1, Nx + 1] +
            #                             u2[i+1,Nx] - 2 * u2[i,Nx] + u2[i-1,Nx]))
            # left
            # u2[i,   0]=a*(-u2[i,1] + 2 * u1[i,1] - u0[i, 1] + 2 * u1[i, 0] - u0[i, 0] +
            #               lambda_v * (u2[i,1] - u0[i, 1] + u0[i, 0]) +
            #               .5 * lambda2 * (u0[i + 1, 0] - 2 * u0[i, 0] + u0[i - 1, 0] +
            #                               u2[i+1,1] - 2 * u2[i,1] + u2[i-1,1]))


