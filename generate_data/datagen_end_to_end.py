import numpy as np
from wave_propagation import pseudo_spectral
import initial_conditions as init_cond
from wave_util import WaveEnergyComponentField_end_to_end, crop_center

def generate_wave_from_medium(input_path, output_path, init_res_f = 128, absorbing_bc = False):

    # b x n_it x n_snaps w x h    ->    b x 200 x 10 x 128 x 128

    ################################### setup ###################################

    res_f = init_res_f

    # parameter setup
    total_time, delta_t_star = 2, .2  #total time of elapses, stepsize per elapse
    f_delta_x = 2.0/128.0   #discretization in spatial (fine discretization, fine solver)
    f_delta_t = f_delta_x / 20  #discretization in time (fine discretization, fine solver)
    n_snaps = round(total_time / delta_t_star)  #number of snapshots

    # data setup
    velocities = np.load(input_path)['wavespeedlist']
    n_it = 10#velocities.shape[0]  # define the amount of data to generate
    centers, widths = np.random.rand(n_it, 2) * 1. - 0.5, 250 + np.random.randn(n_it) * 10

    # tensors for fine solutions in energy components form
    Ux = np.zeros([n_it, n_snaps+1, res_f, res_f])
    Uy = np.zeros([n_it, n_snaps+1, res_f, res_f])
    Utc = np.zeros([n_it, n_snaps + 1, res_f, res_f])

    # tensor for velocity models
    V = np.zeros([n_it, n_snaps+1, res_f, res_f])

    #TODO: double check center cropping and see if it works visualize
    ################################# training #################################

    print("start end to end training data generation, amount of data to generate:", n_it)

    for it in range(n_it):
        print('sample:', it)

        #initialization of wave field
        vel = velocities[it, :, :]
        if absorbing_bc:
            res_f = np.amax(vel) * res_f + 5  # max value the wave can propagate for delta t star, plus rounding errors into account
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res_f), np.linspace(-1, 1, res_f))
        u_elapse, ut_elapse = init_cond.init_cond_gaussian(grid_x, grid_y, widths[it], centers[it])

        # cropping step
        if absorbing_bc:
            u_elapse, ut_elapse = crop_center(u_elapse, init_res_f, init_res_f), crop_center(u_elapse, init_res_f, init_res_f)

        Ux[it, 0, :, :], Uy[it, 0, :, :], Utc[it, 0, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)

        # save velocity model
        V[it,:, :, :] = np.repeat(vel[np.newaxis,:,:], n_snaps+1, axis=0)

        for s in range(1, n_snaps+1):

            # integrate one step delta t star
            # TODO: here is an error since we do not get u_elapse and ut_elapse another grid resolution, we have to change the main function pseudo_spectral for that
            u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)

            # cropping step
            if absorbing_bc:
                u_elapse, ut_elapse = crop_center(u_elapse, init_res_f, init_res_f), crop_center(u_elapse, init_res_f, init_res_f)

            # save current snapshot
            Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)

    np.savez(output_path, vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":
    import sys

    res_f = "128"  # -> will be target resolution
    # res_f = sys.argv[2]

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_"+res_f+".npz",
                              output_path="../data/end_to_end_bp_m_200_"+res_f+"_energytest.npz",
                              res_f=int(res_f), absorbing_bc = True)