import sys
sys.path.append("..")
import numpy as np
from wave_propagation import pseudo_spectral
import initial_conditions as init_cond
from utils_wave import WaveEnergyComponentField_end_to_end, crop_center, start_logger, get_datagen_end_to_end_params
from models.model_utils import get_params
from analysis.visualize_input.visualize_datagen import visualize_wavefield
import logging

# TODO: fix scaler, fix how velocity is cropped, fix crack in surface velocity


def generate_wave_from_medium(input_path, output_path, res = 128, boundary_condition = "absorbing", visualize=False, index="0"):

    # vel: n_it x n_snaps w x h -> 200 x 10 x 128 x 128

    ################################### setup ###################################

    # parameter setup
    start_logger(index=index)
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps, scaler, n_it = get_datagen_end_to_end_params(get_params("0"))

    # data setup
    velocities, n_it, res_padded = init_cond.get_velocities(n_it, input_path, res, boundary_condition)

    # tensors for fine solutions in energy components form
    Ux = np.zeros([n_it, n_snaps + 1, res, res])
    Uy = np.zeros([n_it, n_snaps + 1, res, res])
    Utc = np.zeros([n_it, n_snaps + 1, res, res])

    # tensor for velocity models
    V = np.zeros([n_it, n_snaps+1, res, res])


    ################################# training #################################

    logging.info(" ".join(["start end to end training data generation, amount of data to generate:", str(n_it)]))

    for it in range(n_it):
        logging.info(" ".join(['sample:', str(it)]))

        #initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_elapse, ut_elapse = init_cond.init_cond_gaussian(vel, res, boundary_condition, mode="generate_data", res_padded=res_padded)


        # velocity crop
        if boundary_condition == "absorbing": vel_crop = crop_center(vel, res)
        else: vel_crop = vel
        V[it, :, :, :] = np.repeat(vel_crop[np.newaxis, :, :], n_snaps + 1, axis=0) # save velocity model

        if visualize: visualize_wavefield(u_elapse, ut_elapse, vel = vel, init_res_f=res, frame=True, f_delta_x=f_delta_x, f_delta_t=f_delta_t)

        for s in range(n_snaps+1):
            # integrate delta t star step size n_snaps times

            if boundary_condition == "absorbing":
                # cropping and save current snapshot
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, res), crop_center(ut_elapse, res)
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)

            else:
                # save current snapshot
                Ux[it, s, :, :], Uy[it, s, :, :], Utc[it, s, :, :] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel_crop, f_delta_x)

            if s < n_snaps + 1: u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)

            if visualize: visualize_wavefield(u_elapse, ut_elapse, vel=vel, init_res_f=res, frame=True, f_delta_x=f_delta_x, f_delta_t=f_delta_t)

    np.savez(output_path, vel=V, Ux=Ux, Uy=Uy, Utc=Utc)





if __name__ == "__main__":

    for index in range(3,8):

        generate_wave_from_medium(input_path="../data/crops_bp_m_200_2000.npz",
                                  output_path="../data/end_to_end_bp_m_10_2000_"+str(index)+"_500.npz",
                                  res=128, boundary_condition = "absorbing", visualize = True, index=str(index))



