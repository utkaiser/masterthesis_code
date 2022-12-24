import sys
sys.path.append("..")
import numpy as np
from wave_propagation import pseudo_spectral
import initial_conditions as init_cond
from wave_util import WaveEnergyComponentField_end_to_end, crop_center
from models.model_utils import get_params
from visualize_datagen import visualize_wavefield

def generate_wave_from_medium(input_path, output_path, init_res_f = 128, absorbing_bc = False, visualize=False):

    # vel: n_it x n_snaps w x h -> 200 x 10 x 128 x 128

    ################################### setup ###################################

    # parameter setup
    res_padded = init_res_f
    param_dict = get_params("0")
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps, scaler = \
        param_dict["total_time"], param_dict["delta_t_star"], param_dict["f_delta_x"], param_dict["f_delta_t"], param_dict["n_snaps"], param_dict["res_scaler"]

    # data setup
    # velocities = np.load(input_path)['wavespeedlist']
    n_it = 20 #velocities.shape[0]  # define the amount of data to generate
    velocities = init_cond.diagonal_ray(n_it)

    # tensors for fine solutions in energy components form
    Ux = np.zeros([n_it, n_snaps + 1, init_res_f, init_res_f])
    Uy = np.zeros([n_it, n_snaps + 1, init_res_f, init_res_f])
    Utc = np.zeros([n_it, n_snaps + 1, init_res_f, init_res_f])

    # tensor for velocity models
    V = np.zeros([n_it, n_snaps+1, init_res_f, init_res_f])


    ################################# training #################################

    print("start end to end training data generation, amount of data to generate:", n_it)

    for it in range(n_it):
        print('sample:', it)

        #initialization of wave field
        u_elapse, ut_elapse, res_padded = init_cond.init_cond_gaussian(it, init_res_f, res_padded, absorbing_bc=True)
        vel = velocities[it, :, :]

        # velocity crop
        vel = crop_center(vel, res_padded, res_padded, scaler)
        if absorbing_bc: vel_crop = crop_center(vel, init_res_f, init_res_f, scaler)
        else: vel_crop = vel
        V[it, :, :, :] = np.repeat(vel_crop[np.newaxis, :, :], n_snaps + 1, axis=0) # save velocity model

        if visualize: visualize_wavefield((WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)), vel = vel, init_res_f=init_res_f, frame=True)

        for s in range(n_snaps+1):
            # integrate delta t star step size n_snaps times

            if absorbing_bc:
                # cropping and save current snapshot
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, init_res_f, init_res_f,scaler), crop_center(ut_elapse, init_res_f, init_res_f,scaler)
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)

            else:
                # save current snapshot
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel_crop, f_delta_x)

            if s < n_snaps + 1: u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)

            if visualize: visualize_wavefield((WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel, f_delta_x)), vel=vel, init_res_f=init_res_f, frame=True)

    np.savez(output_path, vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":

    #res_f = ""  # -> will be target resolution
    #res_f = sys.argv[2]

    generate_wave_from_medium(input_path="../data/crops_bp_m_200_2000.npz",
                              output_path="../data/end_to_end_bp_m_20_diagonal_ray.npz",
                              init_res_f=128, absorbing_bc = True, visualize = False)




