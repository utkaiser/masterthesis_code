import torch
import matplotlib.pyplot as plt

def solution_unten(u0, ut0, vel):

    dx = 2.0/128.0
    dt = dx / 20
    cT = .2

    Nt = round(cT/dt)
    Ny, Nx = u0.shape[-1]-1, u0.shape[-2]-1
    c2 = torch.mul(vel,vel)

    lambda_v=abs(dt/dx)
    lambda2=lambda_v**2
    lambdaC2=lambda2*c2

    a=dx/(dx+abs(dt))

    #Euler step to generate u1 from u0 and ut0
    uneg1= u0 - dt * ut0
    u2=u0.clone()
    u1=u0.clone()
    u0=uneg1.clone()

    for k in range(Nt):
        # wave equation update

        u2[1: Ny, 1: Nx] = 2 * u1[1: Ny, 1: Nx]-u0[1: Ny, 1: Nx]+ lambdaC2[1: Ny, 1: Nx]*\
                           (u1[2:Ny+1, 1:Nx] + u1[0: Ny - 1, 1: Nx]+u1[1: Ny, 2: Nx + 1]+u1[1: Ny, 0: Nx - 1]-4 * u1[1: Ny, 1: Nx])

        # absorbing boundary update (Engquist-Majda ABC second order)
        Ny, Nx = Ny - 1, Nx - 1
        u2[-1, 1:Nx+1] = a*(-u2[Ny,1:Nx+1] + 2 * u1[-1,1:Nx+1] - u0[-1,1:Nx+1] + 2*u1[Ny,1:Nx+1] - u0[Ny,1:Nx+1]+
                            lambda_v * (u2[Ny,1:Nx+1] - u0[Ny,1:Nx+1] + u0[-1,1:Nx+1]) +
                            .5 * lambda2 * (u0[-1,2:Nx+2] - 2*u0[-1,1:Nx+1] + u0[-1,0:Nx] +
                                            u2[Ny,2:Nx+2] - 2 * u2[Ny,1:Nx+1] + u2[Ny,0:Nx]))

        u2[0,1:Nx+1] = a*(-u2[1,1:Nx+1]+2*u1[0,1:Nx+1]-u0[0,1:Nx+1]+2*u1[1,1:Nx+1]-u0[1,1:Nx+1]+
                          lambda_v*(u2[1,1:Nx+1]-u0[1,1:Nx+1]+u0[0,1:Nx+1])+
                          .5*lambda2*(u0[0,2:Nx+2] -2*u0[0,1:Nx+1]+u0[0,0:Nx]+
                                      u2[1,2:Nx+2] - 2*u2[1,1:Nx+1]+u2[1,0:Nx]))

        u2[1:Ny+1, -1] = a * (-u2[1:Ny+1, Nx] + 2 * u1[1:Ny+1, Nx] - u0[1:Ny+1, Nx] + 2 * u1[1:Ny+1, Nx + 1] - u0[1:Ny+1, Nx + 1] +
                         lambda_v * (u2[1:Ny+1, Nx] - u0[1:Ny+1, Nx] + u0[1:Ny+1, Nx + 1]) +
                         .5 * lambda2 * (u0[2:Ny+2, Nx + 1] - 2 * u0[1:Ny+1, Nx + 1] + u0[0:Ny, Nx + 1] +
                                         u2[2:Ny+2, Nx] - 2 * u2[1:Ny+1, Nx] + u2[0:Ny, Nx]))

        u2[1:Ny+1, 0] = a * (-u2[1:Ny+1, 1] + 2 * u1[1:Ny+1, 1] - u0[1:Ny+1, 1] + 2 * u1[1:Ny+1, 0] - u0[1:Ny+1, 0] +
                        lambda_v * (u2[1:Ny+1, 1] - u0[1:Ny+1, 1] + u0[1:Ny+1, 0]) +
                        .5 * lambda2 * (u0[2:Ny+2, 0] - 2 * u0[1:Ny+1, 0] + u0[0:Ny, 0] +
                                        u2[2:Ny+2, 1] - 2 * u2[1:Ny+1, 1] + u2[0:Ny, 1]))


        # corners
        u2[-1, 0] = a * (u1[-1, 0] - u2[Ny, 0] + u1[Ny, 0] +
                         lambda_v * (u2[Ny, 0] - u1[-1, 0] + u1[Ny, 0]))
        u2[0, 0] = a * (u1[0, 0] - u2[1, 0] + u1[1, 0] +
                        lambda_v * (u2[1, 0] - u1[0, 0] + u1[1, 0]))
        u2[0, -1] = a * (u1[0, -1] - u2[0, Nx] + u1[0, Nx] +
                         lambda_v * (u2[0, Nx] - u1[0, -1] + u1[0, Nx]))
        u2[-1, -1] = a * (u1[-1, -1] - u2[Ny, -1] + u1[Ny, -1] +
                          lambda_v * (u2[Ny, -1] - u1[-1, -1] + u1[Ny, -1]))

        #update grids
        u, ut= u2.clone(), (u2 - u0) / (2 * dt)
        u0=u1.clone()
        u1=u2.clone()
        Ny, Nx = Ny + 1, Nx + 1

    return u, ut

if __name__ == '__main__':
    from generate_data import wave_propagation as wav
    path = "../analysis/run_2abc/"
    vel = torch.ones(3,128, 128)
    val_a, val_b = torch.meshgrid(torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128))
    u1 = torch.exp(-250 * (torch.mul(val_a,val_a) + torch.mul(val_b, val_b)))
    u = torch.stack((u1,u1,u1), dim=0)
    ut = torch.zeros(3,128, 128)

    for i in range(13):
        plt.imshow(u[0,:,:])
        plt.colorbar()
        plt.show()
        u, ut =  wav.velocity_verlet_tensor(u,ut,boundary_c='absorbing',vel=torch.ones((3,128,128)), dx=2.0/128.0, dt=(2.0/128.0) / 20, delta_t_star=.2)

        #solution_unten(u, ut, vel)