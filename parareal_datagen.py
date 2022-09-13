import numpy as np
from skimage.transform import resize  # for Coarsening
import ParallelCompute as PComp
import WavePostprocess4input as WavePostprocess  #
import WaveUtil
import wave2 as wave2
import wave2_spectral as w2s
import sys
from scipy.io import loadmat
import OPPmodel

def generate_wave_from_medium(input_path, output_path, n_samples_limit = float("inf")):
    """
        generate data pair coarse and fine solutions.
        We first take a velocity sample, then take an initial
        wavefield sample. Then propagate the wavefield using
        the Procrustes parareal scheme, during which the pair
        coarse-fine solutions are computed.
    """

    # Parameter setups
    T = 0.64
    cT = 0.064
    dx = 0.01  # 2.0/128.0
    dt = dx / 20
    pimax = 5

    print(cT)

    ncT = round(T / cT)
    Ny = 256
    Nx = 256
    nx = 64
    ny = 64
    # Coarsening config
    m = 4
    dX = dx * m
    dT = dX / 10

    xx, yy = np.meshgrid(np.linspace(-1, 1, Nx),
                         np.linspace(-1, 1, Ny))
    XX, YY = np.meshgrid(np.linspace(-1, 1, nx),
                         np.linspace(-1, 1, ny))

    velf = np.load(input_path + '.npz')
    #vellist = velf['wavespeedlist']
    vellist = np.zeros([10,xx.shape[0],xx.shape[1]])
    datamat = loadmat('marm1nonsmooth.mat')
    for i in range(10):
       vellist[i,:,:] = resize(datamat['marm1smal'],[256,256])/4.

    # Define the amount of data to generate
    n_trainsamples = vellist.shape[0]
    n_timeslices = pimax * ncT

    # variables for initial conditions
    u_init = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    ut_init = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])

    # variables for initial conditions in energy components form
    U0x = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    U0y = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    Ut0c = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])

    # variables for coarse solutions in energy components form
    Ucx = np.zeros([XX.shape[0], XX.shape[1], n_timeslices * n_trainsamples])
    Ucy = np.zeros([XX.shape[0], XX.shape[1], n_timeslices * n_trainsamples])
    Utc = np.zeros([XX.shape[0], XX.shape[1], n_timeslices * n_trainsamples])
    # variables for fine solutions in energy components form
    Ufx = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    Ufy = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    Utf = np.zeros([xx.shape[0], xx.shape[1], n_timeslices * n_trainsamples])
    # variable for the sampled velocity models
    velsamp = np.zeros([XX.shape[0], YY.shape[1], n_timeslices * n_trainsamples])

    centers1 = np.random.rand(n_trainsamples, 2) * 1. - 0.5
    widths = 250 + np.random.randn(n_trainsamples) * 10

    for j in range(n_trainsamples):
        print('sample', j)
        u_init[:, :, j * n_timeslices], ut_init[:, :, j * n_timeslices] = initCond_ricker(xx, yy, widths[j],
                                                                                          centers1[j, :])
        vel = vellist[j, :, :]
        velX = resize(vel, XX.shape, order=4)

        ######################################################

        up, utp, velX = InitParareal(u_init[:, :, j * n_timeslices], ut_init[:, :, j * n_timeslices], \
                                     vel, dx, dt, cT, dX, dT, T, pimax)

        # Parareal iteration
        for parI in range(pimax - 1):
            #### SUBJECT TO CHANGE TO MULTIPROCESSING
            # Parallel solution
            vx = up[:, :, :, parI]
            vtx = utp[:, :, :, parI]
            print('iteration', parI)

            UcX, UtcX, UfX, UtfX = PComp.ParallelCompute(vx, vtx, vel, velX, dx, dX, dt, dT, cT)
            udx, udy, utdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, velX, dX)
            UcX = resize(UcX, [Ny, Nx], order=4)
            UtcX = resize(UtcX, [Ny, Nx], order=4)
            UcXdx, UcXdy, UtcXdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, vel, dx)
            UfXdx, UfXdy, UtfXdt = WaveUtil.WaveEnergyComponentField(UfX, UtfX, vel, dx)

            ridx = np.arange(j * n_timeslices + parI * ncT, j * n_timeslices + (parI + 1) * ncT)
            Ucx[:, :, ridx] = udx
            Ucy[:, :, ridx] = udy
            Utc[:, :, ridx] = utdt
            Ufx[:, :, ridx] = UfXdx
            Ufy[:, :, ridx] = UfXdy
            Utf[:, :, ridx] = UtfXdt

            velsamp[:, :, ridx] = np.repeat(velX[:, :, np.newaxis], ncT, axis=2)

            if parI == 0:
                P, S, Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), datmode='numpy')
            else:
                P, S, Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), (P, S, Q),
                                                      datmode='numpy')

            # Serial update
            for j in range(ncT - 1):
                w0 = resize(up[:, :, j, parI + 1], [ny, nx], order=4)
                wt0 = resize(utp[:, :, j, parI + 1], [ny, nx], order=4)
                wX, wtX = w2s.wave2(w0, wt0, velX, dX, dT, cT)

                uX, utX = WavePostprocess.ApplyOPP2WaveSol(resize(wX, vel.shape, order=4),
                                                           resize(wtX, vel.shape, order=4), \
                                                           vel, dx, (P, S, Q))

                vX, vtX = WavePostprocess.ApplyOPP2WaveSol(resize(UcX[:, :, j + 1], vel.shape, order=4), \
                                                           resize(UtcX[:, :, j + 1], vel.shape, order=4), vel, dx,
                                                           (P, S, Q))

                up[:, :, j + 1, parI + 1] = UfX[:, :, j + 1] + uX - vX
                utp[:, :, j + 1, parI + 1] = UtfX[:, :, j + 1] + utX - vtX

                ######################################################

    # Save array
    print('Saving data ...')
    np.savez('./data/' + output_path + '.npz', vel=velsamp, Ucx=Ucx, Ucy=Ucy, Utc=Utc, Ufx=Ufx,
             Ufy=Ufy, Utf=Utf)
    print('Saving done.')



def InitParareal(u0, ut0, vel, dx, dt, cT, dX, dT, T, pimax):
    """
    Initial guess in parareal scheme
    """
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T / cT)
    Ny, Nx = vel.shape
    mx = int(dX / dx)
    nx = round(Nx / mx)
    ny = round(Ny / mx)

    velX = resize(vel, [ny, nx], order=4)

    # Store solution at every time slice and parareal iteration
    up = np.zeros([Ny, Nx, ncT, pimax])
    utp = np.zeros([Ny, Nx, ncT, pimax])

    # Set initial condition
    for i in range(pimax):
        up[:, :, 0, i] = u0
        utp[:, :, 0, i] = ut0

    UX = resize(u0, [ny, nx], order=4)
    UtX = resize(ut0, [ny, nx], order=4)

    # Initialize iteration with coarse solution
    for j in range(ncT - 1):
        UX, UtX = wave2.velocity_verlet_time_integrator(UX, UtX, velX, dX, dT, cT)
        up[:, :, j + 1, 0] = resize(UX, [Ny, Nx], order=4)
        utp[:, :, j + 1, 0] = resize(UtX, [Ny, Nx], order=4)

    return up, utp, velX


def initCond(xx, yy, width, center):
    """
    Gaussian pulse wavefield
    u0(x,y)=e−(x2+y2)/σ2,∂tu0(x,y)=0, x,y∈δxZ2 ∩[−1,1)2, 1/σ2 ∼N(250,10).
    """
    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


def initCond_ricker(xx, yy, width, center):
    """
    Ricker pulse wavefield
    """
    u0 = np.exp(-width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2))
    u0 = (1 - 2 * width * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * u0
    u0 = u0 / np.max(np.abs(u0))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


if __name__ == "__main__":
    generate_wave_from_medium(input_path = "mabp4sig_size256cropsM100",
                              output_path = "training_data_12",
                              n_samples_limit = 1)


'''    
import numpy as np
from skimage.transform import resize # for Coarsening
import ParallelCompute as PComp
import WavePostprocess4input as WavePostprocess
import WaveUtil
import wave2
import wave2_spectral as w2s
import OPPmodel

def generate_wave_from_medium(input_path, output_path, n_samples_limit = float("inf")):
    """
        generate data pair coarse and fine solutions.
        generates wave solutions sample from the medium generated in generatecroppedVmodel.py
        We first take a velocity sample (we get from generatecroppedVmodel.py), then take an initial
        wavefield sample. Then propagate the wavefield using
        the Procrustes parareal scheme, during which the pair
        coarse-fine solutions are computed.
    """

    # parameter setup

    T, cT = 0.64, 0.064
    dx = 0.01 #= around 2.0/128.0
    dt = dx/20
    pimax = 5
    ncT = round(T/cT)
    Nx, Ny = 256, 256
    nx, ny = 64, 64

    # Coarsening config
    dX = dx*4 # M=4
    dT = dX/10


    # data setup

    # creates rectangular grid from evenly spaced numbers over a specified interval
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx),
                                 np.linspace(-1, 1, Ny))

    vel_list = np.load(input_path+'.npz')['wavespeedlist']
    n_samples = min(vel_list.shape[0], n_samples_limit) # amount of data to generate
    delta_t = pimax*ncT

    # variables for initial conditions
    u_init = np.zeros([Nx,Nx,delta_t*n_samples])
    ut_init = np.zeros([Nx,Nx,delta_t*n_samples])

    # variables for coarse solutions in energy components form
    Ucx = np.zeros([nx,nx,delta_t*n_samples])
    Ucy = np.zeros([nx,nx,delta_t*n_samples])
    Utc = np.zeros([nx,nx,delta_t*n_samples])

    # variables for fine solutions in energy components form
    Ufx = np.zeros([Nx,Nx,delta_t*n_samples])
    Ufy = np.zeros([Nx,Nx,delta_t*n_samples])
    Utf = np.zeros([Nx,Nx,delta_t*n_samples])

    # variable for the sampled velocity models
    velsamp = np.zeros([nx,ny,delta_t*n_samples])

    centers1 = np.random.rand(n_samples,2)*1.-0.5
    widths = 250+np.random.randn(n_samples)*10

    for j in range(n_samples):
        print("-"*20, 'sample', j, "-"*20)

        #init
        u_init[:,:,j*delta_t], ut_init[:,:,j*delta_t] = initCond(grid_x,grid_y,widths[j],centers1[j,:])
        vel = vel_list[j,:,:]

        up,utp,velX = InitParareal(u_init[:,:,j*delta_t],
                                   ut_init[:,:,j*delta_t],
                                   vel,dx,cT,dX,dT,T,pimax)

        # Parareal iteration
        for i in range(pimax-1):
            print('iteration:', i)

            #### SUBJECT TO CHANGE TO MULTIPROCESSING
            # Parallel solution
            vx = up[:,:,:,i]
            vtx = utp[:,:,:,i]

            UcX,UtcX,UfX,UtfX = PComp.ParallelCompute(vx,vtx,vel,velX,dx,dX,dt,dT,cT)
            udx,udy,utdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, velX, dX)
            UcX = resize(UcX,[Ny,Nx],order=4)
            UtcX = resize(UtcX,[Ny,Nx],order=4)
            UcXdx,UcXdy,UtcXdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, vel, dx)
            UfXdx,UfXdy,UtfXdt = WaveUtil.WaveEnergyComponentField(UfX, UtfX, vel, dx)

            ridx = np.arange(j*delta_t+i*ncT,j*delta_t+(i+1)*ncT)
            Ucx[:,:,ridx] = udx
            Ucy[:,:,ridx] = udy
            Utc[:,:,ridx] = utdt
            Ufx[:,:,ridx] = UfXdx
            Ufy[:,:,ridx] = UfXdy
            Utf[:,:,ridx] = UtfXdt

            velsamp[:,:,ridx] = np.repeat(velX[:,:,np.newaxis],ncT,axis=2)

            if i == 0: P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), datmode='numpy')
            else: P,S,Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), (P, S, Q), datmode='numpy')

            # Serial update
            for j in range(ncT-1):
                w0 = resize(up[:,:,j,i+1],[ny,nx],order=4)
                wt0 = resize(utp[:,:,j,i+1],[ny,nx],order=4)
                wX,wtX = w2s.wave2(w0,wt0,velX,dX,dT,cT)

                uX,utX = WavePostprocess.ApplyOPP2WaveSol(resize(wX,vel.shape,order=4),resize(wtX,vel.shape,order=4),
                                                          vel,dx,(P,S,Q))

                vX,vtX = WavePostprocess.ApplyOPP2WaveSol(resize(UcX[:,:,j+1],vel.shape,order=4),
                                                          resize(UtcX[:,:,j+1],vel.shape,order=4),vel,dx,(P,S,Q))

                up[:,:,j+1,i+1] = UfX[:,:,j+1] + uX - vX
                utp[:,:,j+1,i+1] = UtfX[:,:,j+1] + utX - vtX

    np.savez('./data/'+output_path+'.npz', vel=velsamp, Ucx=Ucx, Ucy=Ucy, Utc=Utc, Ufx=Ufx, Ufy=Ufy, Utf=Utf)


def InitParareal(u0, ut0, vel, dx, cT, dX, dT, T, pimax):
    """
    Initial guess in parareal scheme
    """

    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T / cT)
    Ny, Nx = vel.shape
    mx = int(dX / dx)
    nx = round(Nx / mx)
    ny = round(Ny / mx)

    velX = resize(vel, [ny, nx], order=4)

    # Store solution at every time slice and parareal iteration
    up = np.zeros([Ny, Nx, ncT, pimax])
    utp = np.zeros([Ny, Nx, ncT, pimax])

    # Set initial condition
    for i in range(pimax):
        up[:, :, 0, i] = u0
        utp[:, :, 0, i] = ut0

    UX = resize(u0, [ny, nx], order=4)
    UtX = resize(ut0, [ny, nx], order=4)

    # Initialize iteration with coarse solution
    for j in range(ncT - 1):
        UX, UtX = wave2.velocity_verlet_time_integrator(UX, UtX, velX, dX, dT, cT)
        up[:, :, j + 1, 0] = resize(UX, [Ny, Nx], order=4)
        utp[:, :, j + 1, 0] = resize(UtX, [Ny, Nx], order=4)

    return up, utp, velX


def initCond(grid_x, grid_y, width, center):
    """
    Gaussian pulse wavefield
    u0(x,y)=e−(x2+y2)/σ2,∂tu0(x,y)=0, x,y∈δxZ2 ∩[−1,1)2, 1/σ2 ∼N(250,10).
    """

    u0 = np.exp(-width * ((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2))
    ut0 = np.zeros([np.size(grid_x, axis=1), np.size(grid_y, axis=0)])
    return u0, ut0


def initCond_ricker(grid_x, grid_y, width, center):
    """
    Ricker pulse wavefield
    """

    u0 = np.exp(-width * ((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2))
    u0 = (1 - 2 * width * ((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2)) * u0
    u0 = u0 / np.max(np.abs(u0))
    ut0 = np.zeros([np.size(grid_x, axis=1), np.size(grid_y, axis=0)])
    return u0, ut0


if __name__ == "__main__":

    generate_wave_from_medium(input_path = "mabp4sig_size256cropsM100",
                              output_path = "training_data_12",
                              n_samples_limit = 1)
'''

