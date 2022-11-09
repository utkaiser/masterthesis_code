import torch
import matplotlib.pyplot as plt
import numpy as np

def solution_unten(u0, ut0):

    vel = torch.ones(128,128)
    dx = 2.0/128.0
    dt = dx / 20
    cT = .2

    Nt = round(cT/dt)
    Ny = u0.shape[-1]-1
    Nx = u0.shape[-2]-1
    c2 = torch.mul(vel,vel)

    lambda_v=abs(dt/dx)
    lambda2=lambda_v**2
    lambdaC2=lambda2*c2

    a=dx/(dx+abs(dt))

    #Euler step to generate u1 from u0 and ut0
    uneg1= u0 - dt * ut0
    u2=u0
    u1=u0
    u0=uneg1

    for k in range(Nt):
        # wave equation update
        u2[1:Ny,1:Nx] = 2 *u1[1:Ny,1:Nx] - u0[1:Ny, 1:Nx] + \
                        torch.mul(lambdaC2[1:Ny, 1:Nx],(u1[2:Ny + 1, 1:Nx] +
                                                        u1[0:Ny - 1, 1:Nx] + u1[1:Ny, 2:Nx + 1] + u1[1:Ny, 0:Nx - 1] - 4 * u1[1:Ny, 1:Nx]))

        '''
        # absorbing boundary update (Engquist-Majda ABC second order)
        for j in range(1,Nx+1):
            # bottom
            u2[-1,j]=a*(-u2[Ny,j] + 2 * u1[-1,j] - u0[-1, j] + 2 * u1[Ny, j] - u0[Ny, j] +
                        lambda_v * (u2[Ny,j] - u0[Ny, j] + u0[-1, j]) +
                        .5 * lambda2 * (u0[-1, j + 1] - 2 * u0[-1, j] + u0[-1, j - 1] +
                                        u2[Ny,j+1] - 2 * u2[Ny,j] + u2[Ny,j-1]))
            # top
            u2[0,  j]=a*(-u2[1,j] + 2 * u1[0,j] - u0[0, j] + 2 * u1[1, j] - u0[1, j] +
                         lambda_v * (u2[1,j] - u0[1, j] + u0[0, j]) +
                         .5 * lambda2 * (u0[0, j + 1] - 2 * u0[0, j] + u0[0, j - 1] +
                                         u2[1,j+1] - 2 * u2[1,j] + u2[1,j-1]))


        for i in range(1,Ny+1):
            # right
            u2[i,-1]=a*(-u2[i,Nx] + 2 * u1[i,Nx] - u0[i, Nx] + 2 * u1[i, Nx + 1] - u0[i, Nx + 1] +
                        lambda_v * (u2[i,Nx] - u0[i, Nx] + u0[i, Nx + 1]) +
                        .5 * lambda2 * (u0[i + 1, Nx + 1] - 2 * u0[i, Nx + 1] + u0[i - 1, Nx + 1] +
                                        u2[i+1,Nx] - 2 * u2[i,Nx] + u2[i-1,Nx]))
            # left
            u2[i,   0]=a*(-u2[i,1] + 2 * u1[i,1] - u0[i, 1] + 2 * u1[i, 0] - u0[i, 0] +
                          lambda_v * (u2[i,1] - u0[i, 1] + u0[i, 0]) +
                          .5 * lambda2 * (u0[i + 1, 0] - 2 * u0[i, 0] + u0[i - 1, 0] +
                                          u2[i+1,1] - 2 * u2[i,1] + u2[i-1,1]))


        # borders

        u2[-1, 0] = a * (u1[-1, 0] - u2[Ny - 1, 0] + u1[Ny - 1, 0] +
                         lambda_v * (u2[Ny - 1, 0] - u1[-1, 0] + u1[Ny - 1, 0]))

        u2[0, 0] = a * (u1[0, 0] - u2[1, 0] + u1[1, 0] +
                        lambda_v * (u2[1, 0] - u1[0, 0] + u1[1, 0]))

        u2[0, -1] = a * (u1[0, -1] - u2[0, Nx - 1] + u1[0, Nx - 1] +
                         lambda_v * u2[0, Nx - 1] - u1[0, -1] + u1[0, Nx - 1])

        u2[-1, -1] = a * (u1[-1, -1] - u2[Ny - 1, -1] + u1[Ny - 1, -1] +
                          lambda_v * (u2[Ny - 1, -1] - u1[-1, -1] + u1[Ny - 1, -1]))

        '''
        #update grids
        ut= (u2 - u0) / (2 * dt)
        u=u2

        u0=u1
        u1=u2

    return u, ut

if __name__ == '__main__':
    #from generate_data import wave_propagation as wav

    # val_a, val_b = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128))
    # u = torch.exp(-250 * (torch.mul(val_a,val_a) + torch.mul(val_b, val_b)))
    # ut = torch.zeros((128, 128))
    #
    # for _ in range(8):
    #     plt.imshow(u)
    #     plt.show()
    #     u, ut = solution_unten(u, ut)  # wav.velocity_verlet_tensor(u,ut,boundary_c='periodic',vel=torch.ones((128,128)), dx=2.0/128.0, dt=(2.0/128.0) / 20, delta_t_star=.2)

    pass

