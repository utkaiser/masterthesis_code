import sys
sys.path.append("..")
import numpy as np
from wave_propagation import pseudo_spectral
from initial_conditions import initial_condition_gaussian, get_velocities
from utils_wave import WaveEnergyComponentField_end_to_end, crop_center, get_datagen_end_to_end_params, \
    start_logger_datagen_end_to_end
from models.model_utils import get_params
from analysis.visualize_input.visualize_datagen import visualize_wavefield
import logging


def generate_wave_from_medium(output_path, res, boundary_condition, visualize, index, n_it):

    # parameter setup
    start_logger_datagen_end_to_end(index=index)
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps = get_datagen_end_to_end_params(get_params("0"))

    # data setup
    velocities, n_it, res_padded, output_appendix = get_velocities(n_it, res, boundary_condition)

    # tensors setup
    Ux, Uy, Utc = np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res])
    V = np.zeros([n_it, n_snaps+1, res, res])

    # training
    logging.info(" ".join(["start end to end training data generation, amount of data to generate:", str(n_it)]))

    for it in range(n_it):
        logging.info(" ".join(['sample:', str(it)]))

        #initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_elapse, ut_elapse = initial_condition_gaussian(vel, res, boundary_condition, mode="generate_data", res_padded=res_padded)

        # create and save velocity crop
        vel_crop = crop_center(vel, res, boundary_condition)
        V[it] = np.repeat(vel_crop[np.newaxis, :, :], n_snaps + 1, axis=0)

        if visualize: visualize_wavefield(u_elapse, ut_elapse, vel = vel, init_res_f=res, frame=True, f_delta_x=f_delta_x, f_delta_t=f_delta_t, it=0)

        for s in range(n_snaps+1):
            # integrate delta t star step size n_snaps times

            if boundary_condition == "absorbing":
                # cropping and save current snapshot
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, res), crop_center(ut_elapse, res)
                Ux[it, s], Uy[it, s], Utc[it, s] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)

            else:
                # save current snapshot
                Ux[it, s], Uy[it, s], Utc[it, s] = WaveEnergyComponentField_end_to_end(u_elapse, ut_elapse, vel_crop, f_delta_x)

            if s < n_snaps + 1: u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t, delta_t_star)

            if visualize: visualize_wavefield(u_elapse, ut_elapse, vel=vel, init_res_f=res, frame=True, f_delta_x=f_delta_x, f_delta_t=f_delta_t, it=s+1)

    np.savez(output_path+output_appendix+str(res)+".npz", vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":

    for index in range(1):
        generate_wave_from_medium(output_path = "../data/end_to_end_" + str(index),
                                  res = 128,
                                  boundary_condition = "absorbing",
                                  visualize = True,
                                  index = str(index),
                                  n_it = 10)  # how many data samples to generate



