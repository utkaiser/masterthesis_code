import sys
sys.path.append("../..")
from generate_data.initial_conditions import get_velocities, initial_condition_gaussian
from generate_data.wave_propagation import pseudo_spectral
from models.model_end_to_end import get_model
from models.model_utils import get_params
from generate_data.utils_wave import start_logger_datagen_end_to_end, get_datagen_end_to_end_params, \
    crop_center, WaveEnergyComponentField_end_to_end
import numpy as np
from analysis.visualize_input.visualize_datagen import visualize_wavefield
import logging


def generate_wave_from_medium(output_path, res, boundary_condition, visualize, index, n_it, optimization, n_opti=2):

    # parameter setup
    prefix = "../"
    param_dict = get_params("0")
    start_logger_datagen_end_to_end(index=index, prefix = prefix)
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps = get_datagen_end_to_end_params(param_dict, optimization)
    model = get_model(param_dict, 2, res)

    # data setup
    velocities, n_it, res_padded, output_appendix = get_velocities(n_it, res, boundary_condition, optimization = optimization, prefix = prefix)

    # tensors setup
    Ux, Uy, Utc = np.zeros([n_opti, n_it, n_snaps + 1, res, res]), np.zeros([n_opti, n_it, n_snaps + 1, res, res]), np.zeros([n_opti, n_it, n_snaps + 1, res, res])
    V = np.zeros([n_opti, n_it, n_snaps+1, res, res])

    # training
    logging.info(" ".join(["start end to end training data generation, amount of data to generate:", str(n_it)]))

    for it in range(n_it):
        logging.info(" ".join(['sample:', str(it)]))

        #initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_elapse, ut_elapse = initial_condition_gaussian(vel, res, boundary_condition, mode="generate_data", res_padded=res_padded, optimization=optimization)

        for k in range(n_opti+2):
            # create and save velocity crop
            vel_crop = crop_center(vel, res, boundary_condition)
            V[k, it] = np.repeat(vel_crop[np.newaxis, :, :], n_snaps + 1, axis=0)


            for s in range(n_snaps + 1):  # integrate delta t star step size n_snaps times

                # cropping and save current snapshot
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, res), crop_center(ut_elapse, res)
                Ux[k,it, s], Uy[k,it, s], Utc[k,it, s] = WaveEnergyComponentField_end_to_end(u_elapse_crop,
                                                                                       ut_elapse_crop, vel_crop,
                                                                                       f_delta_x)
                if s < n_snaps + 1:

                    if k == 0:
                        # just model solution
                        pass

                    else:  # k = 1, 2, 3, 4

                        # parareal + model iterations
                        pass

                    if k == n_opti + 2:

                        # fine solver solution
                        u_elapse, ut_elapse = pseudo_spectral(u_elapse, ut_elapse, vel, f_delta_x, f_delta_t,
                                                              delta_t_star)

            if visualize: visualize_wavefield(u_elapse, ut_elapse, vel=vel, init_res_f=res, frame=True, f_delta_x=f_delta_x, f_delta_t=f_delta_t, it=s+1)
    np.savez(output_path+output_appendix+str(res)+"_"+str(optimization)+".npz", vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":

    for index in range(1):
        generate_wave_from_medium(output_path = "../data/end_to_end_" + str(index),
                                  res = 128,
                                  boundary_condition = "absorbing",
                                  visualize = True,
                                  index = str(index),
                                  n_it = 10,  # how many data samples to generate
                                  optimization = "parareal")  # "parareal", "procrustes"



