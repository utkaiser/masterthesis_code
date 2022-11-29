import numpy as np
from wave_propagation import pseudo_spectral, velocity_verlet
import initial_conditions as init_cond
from wave_util import WaveEnergyComponentField_end_to_end, crop_center, WaveEnergyField
from models.model_utils import get_params
from visualize_datagen import visualize_wavefield
import matplotlib.pyplot as plt


def generate_wave_from_medium(input_path, output_path, init_res_f = 128, absorbing_bc = False, visualize=False):

    # vel: n_it x n_snaps w x h -> 200 x 10 x 128 x 128

    ################################### setup ###################################

    # parameter setup
    res_f = init_res_f
    param_dict = get_params("0")
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps, scaler = \
        param_dict["total_time"], param_dict["delta_t_star"], param_dict["f_delta_x"], param_dict["f_delta_t"], param_dict["n_snaps"], param_dict["res_scaler"]

    # data setup
    velocities = np.load(input_path)['wavespeedlist']
    n_it = 10 #velocities.shape[0]  # define the amount of data to generate
    centers, widths = np.random.rand(n_it, 2) * 1. - 0.5, 250 + np.random.randn(n_it) * 10

    # tensors for fine solutions in energy components form
    Ux = np.zeros([n_it, n_snaps + 1, res_f, res_f])
    Uy = np.zeros([n_it, n_snaps + 1, res_f, res_f])
    Utc = np.zeros([n_it, n_snaps + 1, res_f, res_f])

    # tensor for velocity models
    V = np.zeros([n_it, n_snaps+1, res_f, res_f])


    ################################# training #################################

    print("start end to end training data generation, amount of data to generate:", n_it)

    for it in range(n_it):
        print('sample:', it)

        #initialization of wave field
        vel = velocities[it, :, :]
        if absorbing_bc:
            res_f = 300 #math.ceil((init_res_f + math.ceil(np.amax(vel) * delta_t_star * (n_snaps+1) * init_res_f) + 5) / 2.) * 2 # max value the wave can propagate for delta t star, plus rounding errors into account
        widths_scaler = 3+(res_f**1.28/1000) if absorbing_bc else 1
        curr_centers, curr_widths = centers / (res_f / init_res_f), widths * widths_scaler * (res_f / init_res_f)  # scale init condition
        grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, res_f), np.linspace(-1, 1, res_f))
        vel = crop_center(vel, res_f, res_f ,scaler)
        u_elapse, ut_elapse = init_cond.init_cond_gaussian(grid_x, grid_y, curr_widths[it], curr_centers[it])

        # cropping step init
        if absorbing_bc:
            u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, init_res_f, init_res_f,scaler), crop_center(ut_elapse, init_res_f, init_res_f,scaler)
            vel_crop = crop_center(vel, init_res_f, init_res_f,scaler)
            Ux[it, 0, :, :], Uy[it, 0, :, :], Utc[it, 0, :, :] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)
        else:
            vel_crop = vel
            Ux[it, 0, :, :], Uy[it, 0, :, :], Utc[it, 0, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)

        if visualize: visualize_wavefield((WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)), vel = vel, init_res_f=init_res_f, frame=True)
        #if visualize: visualize_wavefield((Ux[it, 0, :, :], Uy[it, 0, :, :], Utc[it, 0, :, :]), vel=vel_crop)

        # save velocity model
        V[it,:, :, :] = np.repeat(vel_crop[np.newaxis,:,:], n_snaps+1, axis=0)

        for s in range(1, n_snaps+1):
            # integrate one step delta t star

            if absorbing_bc:
                # cropping and save current snapshot
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, init_res_f, init_res_f,scaler), crop_center(ut_elapse, init_res_f, init_res_f,scaler)
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)
                if s < n_snaps+1: u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)
            else:
                # save current snapshot
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel_crop, f_delta_x)

            if visualize: visualize_wavefield((WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)), vel=vel, init_res_f=init_res_f, frame=True)

    np.savez(output_path, vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":
    import sys

    #res_f = ""  # -> will be target resolution
    #res_f = sys.argv[2]

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_2000.npz",
                              output_path="../data/end_to_end_bp_m_200_2000.npz",
                              init_res_f=128, absorbing_bc = True, visualize = True)




