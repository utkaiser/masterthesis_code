import numpy as np
from skimage.transform import resize
from wave_propagation import parallel_compute
from wave_propagation import pseudo_spectral
import postprocess_wave as post_wave
import opp_model
import wave_util
import initial_conditions as init_cond

def generate_wave_from_medium(input_path, output_path, res_c = 64, res_f = 128):


    ################################### setup ###################################

    # parameter setup
    total_time, delta_t_star = 2, .2  #total time of elapses, stepsize per elapse
    f_delta_x = 2.0/128.0   #discretization in spatial (fine discretization, fine solver)
    f_delta_t = f_delta_x / 20  #discretization in time (fine discretization, fine solver)
    n_parareal_it = 5  #number of parareal iterations
    n_snaps = round(total_time / delta_t_star)  #number of snapshots
    c_delta_x = f_delta_x * (res_f // res_c)  #coarse fine resolution ratio N_x / n_x; scaling
    c_delta_t = c_delta_x / 10  #discretization in time for course solver, specific number ratio
    n_timeslices = n_parareal_it * n_snaps  #number of communication timestep

    # data setup
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res_f), np.linspace(-1, 1, res_f))
    velocities = np.load(input_path)['wavespeedlist']
    n_it = velocities.shape[0]  # define the amount of data to generate
    centers, widths = np.random.rand(n_it, 2) * 1. - 0.5, 250 + np.random.randn(n_it) * 10

    # tensors for coarse solutions in energy components form
    Ucx = np.zeros([res_c, res_c, n_timeslices * n_it])
    Ucy = np.zeros([res_c, res_c, n_timeslices * n_it])
    Utc = np.zeros([res_c, res_c, n_timeslices * n_it])

    # tensors for fine solutions in energy components form
    Ufx = np.zeros([res_f, res_f, n_timeslices * n_it])
    Ufy = np.zeros([res_f, res_f, n_timeslices * n_it])
    Utf = np.zeros([res_f, res_f, n_timeslices * n_it])

    # tensor for velocity models
    V = np.zeros([res_c, res_c, n_timeslices * n_it])


    ################################# training #################################

    print("start training, amount of data to generate:", n_it)

    for it in range(n_it):
        print('-'*20,'sample', it,'-'*20)

        #initialization of wave field
        u_init, ut_init = init_cond.init_cond_gaussian(grid_x, grid_y, widths[it], centers[it])
        vel, vel_c = velocities[it, :, :], resize(velocities[it, :, :], [res_c, res_c], order=4)

        #integrate initial conditions once using coarse solver/ first guess of parareal scheme
        up, utp = init_cond.first_guess_integration(u_init, ut_init, vel, f_delta_x, f_delta_t,
                                                    delta_t_star, n_snaps, n_parareal_it, res_f)

        # parareal iteration
        for p_it in range(n_parareal_it):

            #get snapshots for current parareal iteration p_it
            u, ut = up[:, :, :, p_it], utp[:, :, :, p_it]

            # get indices for current parareal iteration p_it
            p_idxs = np.arange(it * n_timeslices + p_it * n_snaps,
                               it * n_timeslices + (p_it + 1) * n_snaps)

            # each datasample area to have same velocity model -> fit in nn later to have complete dataset
            V[:, :, p_idxs] = np.repeat(vel_c[:, :, np.newaxis], n_snaps, axis=2)

            # for current parareal iteration p_it compute snapshots of parareal
            UcX, UtcX, UfX, UtfX = parallel_compute(u, ut, vel, vel_c,
                                                    f_delta_x, c_delta_x, f_delta_t, c_delta_t, delta_t_star)

            Ucx[:, :, p_idxs], Ucy[:, :, p_idxs], Utc[:, :, p_idxs] = wave_util.WaveEnergyComponentField(UcX, UtcX, vel_c,
                                                                                                      c_delta_x)
            opp_UcXdx, opp_UcXdy, opp_UtcXdt = wave_util.WaveEnergyComponentField(resize(UcX, [res_f, res_f], order=4),
                                                                                  resize(UtcX, [res_f, res_f], order=4),
                                                                                  vel, f_delta_x)

            UfXdx, UfXdy, UtfXdt = wave_util.WaveEnergyComponentField(UfX, UtfX, vel, f_delta_x)
            Ufx[:, :, p_idxs] = UfXdx
            Ufy[:, :, p_idxs] = UfXdy
            Utf[:, :, p_idxs] = UtfXdt

            if p_it < n_parareal_it - 1: #skips last iteration since this part only influences next iteration
                # compute matrix correction
                if p_it == 0:
                    P, S, Q = opp_model.ProcrustesShiftMap(coarse_dat=(opp_UcXdx, opp_UcXdy, opp_UtcXdt),
                                                           fine_dat=(UfXdx, UfXdy, UtfXdt))
                else:
                    P, S, Q = opp_model.ProcrustesShiftMap(coarse_dat=(opp_UcXdx, opp_UcXdy, opp_UtcXdt),
                                                           fine_dat=(UfXdx, UfXdy, UtfXdt), opmap=(P, S, Q))

                # Serial update, sequential step, compute my solution
                for seq_it in range(1, n_snaps):
                    # another way to compute velocity verlet
                    w0, wt0  = resize(up[:, :, seq_it - 1, p_it + 1], [res_c, res_c], order=4),\
                               resize(utp[:, :, seq_it - 1, p_it + 1], [res_c, res_c], order=4)

                    #requirement of convergence of parareal scheme, solver has to match with earlier parareal
                    wX, wtX = pseudo_spectral(w0, wt0, vel_c, c_delta_x, c_delta_t, delta_t_star)

                    #use computed correct P, S, Q to correct coarse solution
                    uX, utX = post_wave.ApplyOPP2WaveSol(resize(wX, vel.shape, order=4),
                                                         resize(wtX, vel.shape, order=4),
                                                         vel, f_delta_x, (P, S, Q))
                    vX, vtX = post_wave.ApplyOPP2WaveSol(resize(UcX[:, :, seq_it], vel.shape, order=4),
                                                               resize(UtcX[:, :, seq_it], vel.shape, order=4),
                                                               vel, f_delta_x, (P, S, Q))

                    # coupling between fine and coarse solution, add and substract parareal coupling
                    up[:, :, seq_it, p_it + 1] = UfX[:, :, seq_it] + uX - vX
                    utp[:, :, seq_it, p_it + 1] = UtfX[:, :, seq_it] + utX - vtX

    np.savez(output_path, vel=V, Ucx=Ucx, Ucy=Ucy, Utc=Utc, Ufx=Ufx,
             Ufy=Ufy, Utf=Utf)



if __name__ == "__main__":
    import sys

    res_c = "64"
    res_f = "128"

    # res_c = sys.argv[1]
    # res_f = sys.argv[2]

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_"+res_f+".npz",
                              output_path="../data/bp_m_200_"+res_c+"_"+res_f+".npz",
                              res_c=int(res_c),
                              res_f=int(res_f))