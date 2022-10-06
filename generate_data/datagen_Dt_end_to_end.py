import numpy as np
from WaveUtil import initCond, WaveEnergyComponentField
import wave2
import wave2_spectral as w2s
from skimage.transform import resize

def get_Dt():

    # parameter setup
    T, cT = 2, .2  # T time, cT time snapshot T_com in paper
    f_delta_x = 2.0 / 128.0  # .01 #discretization in spatial (fine disc, fine solver)
    f_delta_t = f_delta_x / 20  # discretization in time (fine disc, fine solver)
    pimax = 5  # max number of parareal iteration
    ncT = round(T / cT)  # number of snapshot, =10 right now
    Nx, Ny = 256, 256  # grid resolution fine
    nx, ny = 64, 64  # grid resolution coarse
    sizing = Nx // nx

    # Coarsening config
    c_delta_x = f_delta_x * sizing  # coarse fine resolution ratio N_x / n_x; scaling
    c_delta_t = c_delta_x / 10  # discretization in time for course solver, specific number ratio
    n_timeslices =  ncT  # number of communication timestep, how many samples generated from iteration total number of samples running this code

    # data setup

    velf = np.load("../data/crops_bp_m_200_256.npz")
    vellist = velf['wavespeedlist']
    n_samples = vellist.shape[0]
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    widths = 250 + np.random.randn(n_samples) * 10
    centers1 = np.random.rand(n_samples, 2) * 1. - 0.5

    n_samples = vellist.shape[0]  # define the amount of data to generate
    print("amount of data to generate:", n_samples)

    # variable for the sampled velocity models
    velsamp = np.zeros([nx, ny, n_timeslices * n_samples])
    # variables for coarse solutions in energy components form
    X_U_x = np.zeros([nx, ny, n_timeslices * n_samples])
    X_U_y = np.zeros([nx, ny, n_timeslices * n_samples])
    X_U_c = np.zeros([nx, ny, n_timeslices * n_samples])
    # variables for fine solutions in energy components form
    Y_U_x = np.zeros([Nx, Ny, n_timeslices * n_samples])
    Y_U_y = np.zeros([Nx, Ny, n_timeslices * n_samples])
    Y_U_c = np.zeros([Nx, Ny, n_timeslices * n_samples])

    for j in range(n_samples):
        print('-'*20, 'sample', j, '-'*20)

        #initialization of wave field
        u_prev, ut_prev = initCond(grid_x, grid_y, widths[j], centers1[j, :])
        vel = vellist[j, :, :]
        #velsamp =

        for it in range(1, ncT):

            # fine solver elapse
            ufx, uftx = w2s.wave2(u_prev, ut_prev, vel, c_delta_x, c_delta_t, cT)

            ufx_energy, ufy_energy, uftc_energy = WaveEnergyComponentField(ufx, uftx, vel, f_delta_x)

            X_U_x




            u_prev, ut_prev = ufx, uftx

















