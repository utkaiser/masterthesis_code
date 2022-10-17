import numpy as np
from skimage.transform import resize
import parallel_compute as PComp
import wave_propagation as wave2
import wave_propagation_spectral as w2s
import opp_model
import wave_util

# import sys
# import matplotlib.pyplot as plt
# import torch

def generate_wave_from_medium(input_path, output_path, resolution_f = 128, resolution_c = 64):
    """
        generate data pair coarse and fine solutions.
        We first take a velocity sample, then take an initial
        wavefield sample. Then propagate the wavefield using
        the Procrustes parareal scheme, during which the pair
        coarse-fine solutions are computed.
    """

    #TODO: ridx not correct, change of n_parareal_it is wrong, fix this
    #TODO: .08 of inits are wrong, why? is that initial condition

    # parameter setup
    total_time, delta_t_star = 2, .2  #total_time time, delta_t_star time snapshot T_com in paper
    f_delta_x = 2.0/128.0   #discretization in spatial (fine discretization, fine solver)
    f_delta_t = f_delta_x / 20  #discretization in time (fine discretization, fine solver)
    n_parareal_it = 5  #max number of parareal iteration
    n_snapshots = round(total_time / delta_t_star)  #number of snapshot
    c_delta_x = f_delta_x * (resolution_f // resolution_c)  #coarse fine resolution ratio N_x / n_x; scaling
    c_delta_t = c_delta_x / 10  #discretization in time for course solver, specific number ratio
    n_timeslices = n_parareal_it * n_snapshots  #number of communication timestep

    # data setup
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, resolution_f), np.linspace(-1, 1, resolution_f))
    vel_list = np.load(input_path)['wavespeedlist']
    n_samples = 10 #vel_list.shape[0]  # define the amount of data to generate
    print("amount of data to generate:", n_samples)
    centers1 = np.random.rand(n_samples, 2) * 1. - 0.5
    widths = 250 + np.random.randn(n_samples) * 10

    # variables for coarse solutions in energy components form
    Ucx = np.zeros([resolution_c, resolution_c, n_timeslices * n_samples])
    Ucy = np.zeros([resolution_c, resolution_c, n_timeslices * n_samples])
    Utc = np.zeros([resolution_c, resolution_c, n_timeslices * n_samples])

    # variables for fine solutions in energy components form
    Ufx = np.zeros([resolution_f, resolution_f, n_timeslices * n_samples])
    Ufy = np.zeros([resolution_f, resolution_f, n_timeslices * n_samples])
    Utf = np.zeros([resolution_f, resolution_f, n_timeslices * n_samples])

    V = np.zeros([resolution_c,resolution_c, n_timeslices * n_samples])  #sampled velocity models


    for it in range(n_samples):
        print('-'*20,'sample', it,'-'*20)

        #initialization of wave field
        u_init, ut_init = wave_util.initCond(grid_x, grid_y, widths[it], centers1[it, :])
        vel, vel_c = vel_list[it, :, :], resize(vel_list[it, :, :], [resolution_c, resolution_c], order=4)

        #integrate initial conditions once using coarse solver/ first guess of parareal scheme
        up, utp = init_parareal(u_init, ut_init, vel, f_delta_x,f_delta_t, delta_t_star, n_snapshots,
                                n_parareal_it, resolution_f)

        # parareal iteration
        for i in range(n_parareal_it - 1):

            #get snapshots for current elapse i
            #TODO: understand why up[...] and how connected to ParallelSyncCompute
            vx, vtx = up[:, :, :, i], utp[:, :, :, i]

            # get indices for current elapse i
            ridx = np.arange(it * n_timeslices + i * n_snapshots, it * n_timeslices + (i + 1) * n_snapshots)
            print(ridx)

            # each datasample area to have same velocity model -> fit in nn later to have complete dataset
            V[:, :, ridx] = np.repeat(vel_c[:, :, np.newaxis], n_snapshots, axis=2)

            #way to compute the solution, solve fine and coarse solution using velocity etc., for current solution/iteration
            UcX, UtcX, UfX, UtfX = PComp.ParallelSyncCompute(vx, vtx, vel, vel_c, f_delta_x, c_delta_x, f_delta_t, c_delta_t, delta_t_star) #propagation of wave field for whole interval

            Ucx[:, :, ridx], Ucy[:, :, ridx], Utc[:, :, ridx] = wave_util.WaveEnergyComponentField(UcX, UtcX, vel_c, c_delta_x) #convert above into energy component
            UcXdx, UcXdy, UtcXdt = wave_util.WaveEnergyComponentField(resize(UcX, [resolution_f, resolution_f], order=4),
                                                                      resize(UtcX, [resolution_f, resolution_f], order=4),
                                                                      vel, f_delta_x)

            UfXdx, UfXdy, UtfXdt = wave_util.WaveEnergyComponentField(UfX, UtfX, vel, f_delta_x)
            Ufx[:, :, ridx] = UfXdx
            Ufy[:, :, ridx] = UfXdy
            Utf[:, :, ridx] = UtfXdt

            # compute matrix correction to correct my solution later
            if i == 0: P, S, Q = opp_model.ProcrustesShiftMap(i, coarse_dat=(UcXdx, UcXdy, UtcXdt), fine_dat=(UfXdx, UfXdy, UtfXdt), vel=vel, datmode='numpy')
            else: P, S, Q = opp_model.ProcrustesShiftMap(i, coarse_dat=(UcXdx, UcXdy, UtcXdt), fine_dat=(UfXdx, UfXdy, UtfXdt), opmap=(P, S, Q), vel=vel, datmode='numpy')

            # Serial update, sequential step, compute my solution
            for ppp in range(n_snapshots - 1):
                # another way to compute velocity verlet
                w0 = resize(up[:, :, ppp, i + 1], [resolution_c, resolution_c], order=4)
                wt0 = resize(utp[:, :, ppp, i + 1], [resolution_c, resolution_c], order=4)

                #requirement of convergence of parareal scheme (convergence to exact/ reference wave solution)
                wX, wtX = w2s.wave2s(w0, wt0, vel_c, c_delta_x, c_delta_t, delta_t_star) #solver has to match with parareal compute PComp.ParallelCompute

                #use computed correct P, S, Q to correct coarse solution
                uX, utX = wave_postprocess.ApplyOPP2WaveSol(resize(wX, vel.shape, order=4),
                                                           resize(wtX, vel.shape, order=4),
                                                           vel, f_delta_x, (P, S, Q))
                vX, vtX = wave_postprocess.ApplyOPP2WaveSol(resize(UcX[:, :, ppp + 1], vel.shape, order=4),
                                                           resize(UtcX[:, :, ppp + 1], vel.shape, order=4),
                                                           vel, f_delta_x, (P, S, Q))

                # coupling between fine and coarse solution, add and substract parareal coupling
                up[:, :, ppp + 1, i + 1] = UfX[:, :, ppp + 1] + uX - vX
                utp[:, :, ppp + 1, i + 1] = UtfX[:, :, ppp + 1] + utX - vtX

        #print(np.count_nonzero(V)/128**2)
        # for i in range(V):
        #     if V

    np.savez(output_path, vel=V, Ucx=Ucx, Ucy=Ucy, Utc=Utc, Ufx=Ufx,
             Ufy=Ufy, Utf=Utf)
    ''''''


def init_parareal(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star, n_snapshots, n_parareal_it,
                  resolution_f):
    """
    initial guess in parareal scheme
    """

    # store solution at every time slice and parareal iteration
    up = np.zeros([resolution_f, resolution_f, n_snapshots, n_parareal_it])
    utp = np.zeros([resolution_f, resolution_f, n_snapshots, n_parareal_it])

    # set initial condition
    for i in range(n_parareal_it):
        up[:, :, 0, i], utp[:, :, 0, i] = u_elapse, ut_elapse  # initialize condition u_0, ut_0

    # initialize iteration with fine solution
    for j in range(1, n_snapshots):
        UX, UtX = wave2.velocity_verlet_time_integrator(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)
        up[:, :, j, 0], utp[:, :, j, 0] = UX, UtX

    return up, utp



if __name__ == "__main__":
    #n = "4" #sys.argv[1]
    #print("start training for", n)

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_128.npz",
                              output_path="../data/bp_m_test.npz")