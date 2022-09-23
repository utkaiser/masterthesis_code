import numpy as np
from skimage.transform import resize # for Coarsening
import wave2
import wave2_spectral as w2s
# import matplotlib.pyplot as plt
# from WaveUtil import WaveEnergyField

def ParallelCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):

    ncT = v.shape[2]
    ny,nx = velX.shape
    Ny,Nx = vel.shape

    # Allocate arrays for output
    uf = np.zeros([Ny,Nx,ncT])
    utf = np.zeros([Ny,Nx,ncT])
    uc = np.zeros([ny,nx,ncT])
    utc = np.zeros([ny,nx,ncT])

    for j in range(ncT-1):
        ucx, utcx = wave2.velocity_verlet_time_integrator(
            resize(v[:, :, j], [ny, nx], order=4), resize(vt[:, :, j], [ny, nx], order=4),
            velX, dX, dT, cT
        )

        uc[:, :, j + 1], utc[:, :, j + 1] = ucx, utcx  # resize(ucx,[Ny,Nx],order=4), resize(utcx,[Ny,Nx],order=4)
        a, b = w2s.wave2(v[:, :, j], vt[:, :, j], vel, dx, dt, cT)
        uf[:, :, j + 1], utf[:, :, j + 1] = a, b

        # f, ax = plt.subplots(1, 2)
        # f.set_figheight(5)
        # f.set_figwidth(15)
        #
        # res1 = WaveEnergyField(ucx, utcx, resize(vel,[64,64], order = 4), dX) * dX * dX  # coarse
        # res2 = WaveEnergyField(a, b, vel, dx) * dx * dx  # fine
        # ax[0].imshow(res1, vmax=0.005, extent=(-1, 1, -1, 1))
        # ax[1].imshow(res2, vmax=0.005, extent=(-1, 1, -1, 1))

        #plt.show()

    return uc,utc,uf,utf


def ParallelSyncCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):

    ncT = v.shape[2]
    ny,nx = velX.shape
    Ny,Nx = vel.shape

    # Allocate arrays for output
    uf = np.zeros([Ny,Nx,ncT])
    utf = np.zeros([Ny,Nx,ncT])
    uc = np.zeros([ny,nx,ncT])
    utc = np.zeros([ny,nx,ncT])

    # Parallel loop, each rhs is independent of lhs
    for j in range(ncT):
        ucx,utcx = w2s.wave2s(resize(v[:,:,j],[ny,nx],order=4),resize(vt[:,:,j],[ny,nx],order=4),\
                                    velX,dX,dT,cT)
        uc[:,:,j] = ucx #resize(ucx,[Ny,Nx],order=4)
        utc[:,:,j] = utcx #resize(utcx,[Ny,Nx],order=4)

        uf[:,:,j],utf[:,:,j] = w2s.wave2s(v[:,:,j],vt[:,:,j],vel,dx,dt,cT)

    return uc,utc,uf,utf


