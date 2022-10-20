import numpy as np
from wave_propagation import pseudo_spectral
import initial_conditions as init_cond

def generate_wave_from_medium(input_path, output_path, res_f = 128):


    # b x n_it x n_snaps w x h    ->    b x 200 x 10 x 128 x 128

    ################################### setup ###################################

    # parameter setup
    total_time, delta_t_star = 2, .2  #total time of elapses, stepsize per elapse
    f_delta_x = 2.0/128.0   #discretization in spatial (fine discretization, fine solver)
    f_delta_t = f_delta_x / 20  #discretization in time (fine discretization, fine solver)
    n_snaps = round(total_time / delta_t_star)  #number of snapshots

    # data setup
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res_f), np.linspace(-1, 1, res_f))
    velocities = np.load(input_path)['wavespeedlist']
    n_it = 10 #velocities.shape[0]  # define the amount of data to generate
    centers, widths = np.random.rand(n_it, 2) * 1. - 0.5, 250 + np.random.randn(n_it) * 10

    # tensors for fine solutions in energy components form
    U = np.zeros([n_it, n_snaps+1, res_f, res_f])
    Ut = np.zeros([n_it, n_snaps+1, res_f, res_f])

    # tensor for velocity models
    V = np.zeros([n_it, n_snaps+1, res_f, res_f])


    ################################# training #################################

    print("start end to end training data generation, amount of data to generate:", n_it)

    for it in range(n_it):
        print('sample:', it)

        #initialization of wave field
        u_elapse, ut_elapse = init_cond.init_cond_gaussian(grid_x, grid_y, widths[it], centers[it])
        U[it, 0, :, :], Ut[it, 0, :, :] = u_elapse, ut_elapse
        vel = velocities[it, :, :]

        # save velocity model
        V[it,:, :, :] = np.repeat(vel[np.newaxis,:,:], n_snaps+1, axis=0)

        for s in range(1, n_snaps+1):

            # integrate one step delta t star
            u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)

            # save current snapshot
            U[it, s, :, :], Ut[it, s, :, :] = u_elapse, ut_elapse

    np.savez(output_path, vel=V, U=U, Ut=Ut)


if __name__ == "__main__":
    import sys

    res_f = "128" #  -> will be target resolution
    # res_f = sys.argv[2]

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_"+res_f+".npz",
                              output_path="../data/end_to_end_bp_m_200_"+res_f+"_test.npz",
                              res_f=int(res_f))