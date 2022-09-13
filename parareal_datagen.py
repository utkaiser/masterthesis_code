import numpy as np
from skimage.transform import resize  # for coarsening
import ParallelCompute as PComp
import WavePostprocess4input as WavePostprocess  #
import WaveUtil
import wave2 as wave2
import wave2_spectral as w2s
import OPPmodel

def generate_wave_from_medium(input_path, output_path):
    """
        generate data pair coarse and fine solutions.
        We first take a velocity sample, then take an initial
        wavefield sample. Then propagate the wavefield using
        the Procrustes parareal scheme, during which the pair
        coarse-fine solutions are computed.
    """

    # parameter setup

    T, cT = 0.64, 0.064 #T time, cT time snapshot T_com in paper
    dx = 0.01  # 2.0/128.0 #discretization in spatial (fine disc, fine solver)
    dt = dx / 20 #discretization in time (fine disc, fine solver)
    pimax = 5 #max number of parareal iteration
    ncT = round(T / cT) #number of snapshot =10 right now
    Nx, Ny = 256, 256 #grid resolution fine
    nx, ny = 64, 64 #grid resolution coarse
    # Coarsening config
    dX = dx * 4 #coarse fine resolution ratio N_x / n_x; scaling
    dT = dX / 10 #discretization in time for course solver, specific number ratio
    delta_t = pimax * ncT #number of communication timestep n_timeslices how many samples generated from iteration total number of samples running this code


    # data setup

    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx),
                         np.linspace(-1, 1, Ny))
    velf = np.load(input_path + '.npz')
    vellist = velf['wavespeedlist']
    n_samples = vellist.shape[0] # define the amount of data to generate

    # variables for initial conditions
    u_init = np.zeros([Nx, Nx, delta_t * n_samples])
    ut_init = np.zeros([Nx, Nx, delta_t * n_samples])

    # variables for coarse solutions in energy components form
    Ucx = np.zeros([nx, ny, delta_t * n_samples])
    Ucy = np.zeros([nx, ny, delta_t * n_samples])
    Utc = np.zeros([nx, ny, delta_t * n_samples])
    # variables for fine solutions in energy components form
    Ufx = np.zeros([Nx, Ny, delta_t * n_samples])
    Ufy = np.zeros([Nx, Ny, delta_t * n_samples])
    Utf = np.zeros([Nx, Ny, delta_t * n_samples])
    # variable for the sampled velocity models
    velsamp = np.zeros([nx,ny, delta_t * n_samples])

    centers1 = np.random.rand(n_samples, 2) * 1. - 0.5
    widths = 250 + np.random.randn(n_samples) * 10

    for j in range(n_samples):
        print('-'*20, 'sample', j, '-'*20)

        #initialization of  wave field
        u_init[:, :, j * delta_t], ut_init[:, :, j * delta_t] = initCond_ricker(grid_x, grid_y, widths[j], centers1[j, :]) #p.20
        vel = vellist[j, :, :]

        #integrate initial conditions once using coarse solver/ first guess of parareal scheme
        up, utp, velX = InitParareal(u_init[:, :, j * delta_t], ut_init[:, :, j * delta_t],
                                     vel, dx, cT, dX, dT, T, pimax)

        # parareal iteration
        for i in range(pimax - 1):

            #approximation / preparation of parareal scheme
            #### SUBJECT TO CHANGE TO MULTIPROCESSING #speeding up algorithm
            # Parallel solution, illustrated in Fig. 4
            vx = up[:, :, :, i]
            vtx = utp[:, :, :, i]

            #way to compute the solution, solve fine and coarse solution using velocity etc., for current solution/iteration
            UcX, UtcX, UfX, UtfX = PComp.ParallelCompute(vx, vtx, vel, velX, dx, dX, dt, dT, cT) #propagation of wave field for whole interval
            udx, udy, utdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, velX, dX) #convert above into energy component
            UcX = resize(UcX, [Ny, Nx], order=4)
            UtcX = resize(UtcX, [Ny, Nx], order=4)
            UcXdx, UcXdy, UtcXdt = WaveUtil.WaveEnergyComponentField(UcX, UtcX, vel, dx)
            UfXdx, UfXdy, UtfXdt = WaveUtil.WaveEnergyComponentField(UfX, UtfX, vel, dx)

            #saving solutions in tensor
            ridx = np.arange(j * delta_t + i * ncT, j * delta_t + (i + 1) * ncT)
            Ucx[:, :, ridx] = udx
            Ucy[:, :, ridx] = udy
            Utc[:, :, ridx] = utdt
            Ufx[:, :, ridx] = UfXdx
            Ufy[:, :, ridx] = UfXdy
            Utf[:, :, ridx] = UtfXdt

            #each datasample area to have same velocity model -> fit in nn later to have complete dataset
            velsamp[:, :, ridx] = np.repeat(velX[:, :, np.newaxis], ncT, axis=2)

            #idea: compute matrix correction to correct my solution later
            if i == 0:
                P, S, Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), datmode='numpy')
            else:
                P, S, Q = OPPmodel.ProcrustesShiftMap((UcXdx, UcXdy, UtcXdt), (UfXdx, UfXdy, UtfXdt), (P, S, Q),datmode='numpy')

            # Serial update, sequential step, compute my solution
            for j in range(ncT - 1):
                # another way to compute velocity verlet
                w0 = resize(up[:, :, j, i + 1], [ny, nx], order=4)
                wt0 = resize(utp[:, :, j, i + 1], [ny, nx], order=4)

                #requirement of convergence of parareal scheme (convergence to exact/ reference wave solution)
                wX, wtX = w2s.wave2(w0, wt0, velX, dX, dT, cT) #solver has to match with parareal compute PComp.ParallelCompute line 22

                #use computed correct P, S, Q to correct coarse solution
                uX, utX = WavePostprocess.ApplyOPP2WaveSol(resize(wX, vel.shape, order=4),
                                                           resize(wtX, vel.shape, order=4),
                                                           vel, dx, (P, S, Q))

                vX, vtX = WavePostprocess.ApplyOPP2WaveSol(resize(UcX[:, :, j + 1], vel.shape, order=4),
                                                           resize(UtcX[:, :, j + 1], vel.shape, order=4), vel, dx,
                                                           (P, S, Q))

                # coupling between fine and coarse solution, add and substract parareal coupling, eq. 26
                up[:, :, j + 1, i + 1] = UfX[:, :, j + 1] + uX - vX
                utp[:, :, j + 1, i + 1] = UtfX[:, :, j + 1] + utX - vtX


    np.savez('./data/' + output_path + '.npz', vel=velsamp, Ucx=Ucx, Ucy=Ucy, Utc=Utc, Ufx=Ufx,
             Ufy=Ufy, Utf=Utf)



def InitParareal(u0, ut0, vel, dx, cT, dX, dT, T, pimax):
    """
    Initial guess in parareal scheme
    not in paper, guess solution
    """
    # Number of time slices - fine and coarse propagators communicate
    ncT = round(T / cT)
    Ny, Nx = vel.shape
    mx = int(dX / dx)
    nx = round(Nx / mx)
    ny = round(Ny / mx)

    velX = resize(vel, [ny, nx], order=4)

    # Store solution at every time slice and parareal iteration
    up = np.zeros([Ny, Nx, ncT, pimax]) #variable to store all the sample, init as zero
    utp = np.zeros([Ny, Nx, ncT, pimax])

    # Set initial condition
    for i in range(pimax):
        up[:, :, 0, i] = u0
        utp[:, :, 0, i] = ut0

    #initialize condition using coarse solver (converting fine grid condition), downsampling
    UX = resize(u0, [ny, nx], order=4)
    UtX = resize(ut0, [ny, nx], order=4)

    # Initialize iteration with coarse solution
    for j in range(ncT - 1):
        UX, UtX = wave2.velocity_verlet_time_integrator(UX, UtX, velX, dX, dT, cT)
        up[:, :, j + 1, 0] = resize(UX, [Ny, Nx], order=4) #coarse comp. to fine coarse -> upsampling
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
                              output_path = "training_data_12")