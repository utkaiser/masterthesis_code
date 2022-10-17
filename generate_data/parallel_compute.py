import numpy as np
from skimage.transform import resize # for Coarsening
import generate_data.wave_propagation as wave2
import generate_data.wave_propagation_spectral as w2s

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
        uc[:, :, j + 1], utc[:, :, j + 1] = ucx, utcx

        a, b = w2s.wave2(v[:, :, j], vt[:, :, j], vel, dx, dt, cT)
        uf[:, :, j + 1], utf[:, :, j + 1] = a, b

    return uc,utc,uf,utf


def ParallelSyncCompute(v,vt,vel,velX,dx,dX,dt,dT,cT):

    n_snapshots = v.shape[2]
    resolution_c, resolution_f = velX.shape[0], vel.shape[0]

    # pre-allocate arrays for output
    uf = np.zeros([resolution_f,resolution_f,n_snapshots])
    utf = np.zeros([resolution_f,resolution_f,n_snapshots])
    uc = np.zeros([resolution_c,resolution_c,n_snapshots])
    utc = np.zeros([resolution_c,resolution_c,n_snapshots])

    # parallel loop, each rhs is independent of lhs
    for j in range(n_snapshots):
        uc[:,:,j], utc[:,:,j] = wave2.velocity_verlet_time_integrator(resize(v[:,:,j],[resolution_c,resolution_c],order=4),
                                                                      resize(vt[:,:,j],[resolution_c,resolution_c],order=4),
                                                                      velX,dX,dT,cT)

        uf[:,:,j],utf[:,:,j] = w2s.wave2s(v[:,:,j],vt[:,:,j],vel,dx,dt,cT)

    return uc,utc,uf,utf


