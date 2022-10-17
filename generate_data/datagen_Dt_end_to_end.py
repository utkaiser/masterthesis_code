import numpy as np
from wave_util import initCond
import wave_propagation_spectral as w2s

def datagen_Dt(input_path, output_path):

    # parameter setup
    T, cT = 2, .2  # T time, cT time snapshot T_com in paper
    f_delta_x = 2.0 / 128.0  # .01 #discretization in spatial (fine disc, fine solver)
    f_delta_t = f_delta_x / 20  # discretization in time (fine disc, fine solver)
    ncT = round(T / cT)  # number of snapshot, =10 right now
    Nx, Ny = 128, 128  # grid resolution fine
    n_timeslices =  ncT  # number of communication timestep, how many samples generated from iteration total number of samples running this code

    # data setup
    velf = np.load(input_path)
    vellist = velf['wavespeedlist']
    n_samples = vellist.shape[0]
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    widths = 250 + np.random.randn(n_samples) * 10
    centers1 = np.random.rand(n_samples, 2) * 1. - 0.5
    n_samples = vellist.shape[0]  # define the amount of data to generate
    print("amount of data to generate:", n_samples)

    # variable for the sampled velocity models
    velsamp = np.zeros([Nx, Ny, n_timeslices * n_samples])

    # variables for prev solutions in physical components
    X_U = np.zeros([Nx, Ny, n_timeslices * n_samples])
    X_Ut = np.zeros([Nx, Ny, n_timeslices * n_samples])

    # variables for next fine solutions in physical components
    Y_U = np.zeros([Nx, Ny, n_timeslices * n_samples])
    Y_Ut = np.zeros([Nx, Ny, n_timeslices * n_samples])

    for j in range(n_samples):
        print('-'*20, 'sample', j, '-'*20)

        u_prev, ut_prev = initCond(grid_x, grid_y, widths[j], centers1[j, :]) #initialization of wave field
        vel = vellist[j, :, :] #get velocity model for curr iteration j
        ridx = np.arange(j * n_timeslices, (j+1) * n_timeslices) #get range of indices for curr iteration j
        velsamp[:, :, ridx] = np.repeat(vel[:, :, np.newaxis], ncT, axis=2) #save velocity for curr iteration j

        for it in range(ncT):

            #save input (which is previous solution)
            X_U[:,:,ridx[it]], X_Ut[:,:,ridx[it]] = u_prev, ut_prev

            # fine solver elapse; target
            ufx, uftx = w2s.wave2(u_prev, ut_prev, vel, f_delta_x, f_delta_t, cT)
            Y_U[:,:,ridx[it]], Y_Ut[:,:,ridx[it]] = ufx, uftx #save target

            u_prev, ut_prev = ufx, uftx

    np.savez(output_path, vel=velsamp, X_U=X_U, X_Ut=X_Ut, Y_U=Y_U, Y_Ut=Y_Ut)


def datagen_Dtp(input_path, output_path):

    #TODO: change code below to parareal scheme

    # parameter setup
    T, cT = 2, .2  # T time, cT time snapshot T_com in paper
    f_delta_x = 2.0 / 128.0  # .01 #discretization in spatial (fine disc, fine solver)
    f_delta_t = f_delta_x / 20  # discretization in time (fine disc, fine solver)
    ncT = round(T / cT)  # number of snapshot, =10 right now
    Nx, Ny = 128, 128  # grid resolution fine
    n_timeslices =  ncT  # number of communication timestep, how many samples generated from iteration total number of samples running this code

    # data setup
    velf = np.load(input_path)
    vellist = velf['wavespeedlist']
    n_samples = vellist.shape[0]
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    widths = 250 + np.random.randn(n_samples) * 10
    centers1 = np.random.rand(n_samples, 2) * 1. - 0.5
    n_samples = vellist.shape[0]  # define the amount of data to generate
    print("amount of data to generate:", n_samples)

    # variable for the sampled velocity models
    velsamp = np.zeros([Nx, Ny, n_timeslices * n_samples])

    # variables for prev solutions in physical components
    X_U = np.zeros([Nx, Ny, n_timeslices * n_samples])
    X_Ut = np.zeros([Nx, Ny, n_timeslices * n_samples])

    # variables for next fine solutions in physical components
    Y_U = np.zeros([Nx, Ny, n_timeslices * n_samples])
    Y_Ut = np.zeros([Nx, Ny, n_timeslices * n_samples])

    for j in range(n_samples):
        print('-'*20, 'sample', j, '-'*20)

        u_prev, ut_prev = initCond(grid_x, grid_y, widths[j], centers1[j, :]) #initialization of wave field
        vel = vellist[j, :, :] #get velocity model for curr iteration j
        ridx = np.arange(j * n_timeslices, (j+1) * n_timeslices) #get range of indices for curr iteration j
        velsamp[:, :, ridx] = np.repeat(vel[:, :, np.newaxis], ncT, axis=2) #save velocity for curr iteration j

        for it in range(ncT):

            #save input (which is previous solution)
            X_U[:,:,ridx[it]], X_Ut[:,:,ridx[it]] = u_prev, ut_prev

            # fine solver elapse; target
            ufx, uftx = w2s.wave2(u_prev, ut_prev, vel, f_delta_x, f_delta_t, cT)
            Y_U[:,:,ridx[it]], Y_Ut[:,:,ridx[it]] = ufx, uftx #save target

            u_prev, ut_prev = ufx, uftx

    np.savez(output_path, vel=velsamp, X_U=X_U, X_Ut=X_Ut, Y_U=Y_U, Y_Ut=Y_Ut)



if __name__ == '__main__':
    datagen_Dt(input_path="../data/crops_bp_m_200_128.npz",
           output_path="../data/Dt_end_to_end_bp_m_200_128_psm.npz")














