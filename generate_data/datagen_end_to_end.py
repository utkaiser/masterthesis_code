import sys

from models.optimization_models.parallel_scheme import one_iteration_pseudo_spectral_tensor

sys.path.append("..")
import torch
import numpy as np
from wave_propagation import pseudo_spectral_tensor
from initial_conditions import initial_condition_gaussian, get_velocities
from utils_wave import WaveEnergyComponentField_end_to_end, crop_center, \
    start_logger_datagen_end_to_end, WaveSol_from_EnergyComponent, WaveSol_from_EnergyComponent_tensor
from models.model_utils import get_params
from analysis.visualize_input.visualize_datagen import visualize_wavefield
import logging


def generate_wave_from_medium(output_path, res, boundary_condition, visualize, index, n_it, optimization):

    # parameter setup
    start_logger_datagen_end_to_end(index=index)
    param_dict = get_params("0")
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps = param_dict["total_time"], param_dict["delta_t_star"], param_dict["f_delta_x"], param_dict["f_delta_t"], \
           param_dict["n_snaps"]

    # data setup
    velocities, n_it, res_padded, output_appendix = get_velocities(n_it, res, boundary_condition, optimization=optimization)

    # tensors setup
    Ux, Uy, Utc = np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res])
    V = np.zeros([n_it, n_snaps+1, res, res])

    # training
    logging.info(" ".join(["start end to end training data generation, amount of data to generate:", str(n_it)]))

    for it in range(n_it):
        logging.info(" ".join(['sample:', str(it)]))

        #initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_n = initial_condition_gaussian(torch.from_numpy(vel), res, boundary_condition, res_padded, optimization=optimization, mode="parareal")  # 1 x 3 x 256 x 256

        # create and save velocity crop
        vel_crop = crop_center(vel, res, boundary_condition)
        V[it] = np.repeat(vel_crop[np.newaxis, :, :], n_snaps + 1, axis=0)

        for s in range(n_snaps+1):
            # integrate delta t star step size n_snaps times

            if visualize:
                u_elapse, ut_elapse = WaveSol_from_EnergyComponent_tensor(u_n[:,0], u_n[:,1], u_n[:,2], torch.from_numpy(vel), 2. / 128.,
                                                                          torch.sum(torch.sum(torch.sum(u_n[:,0]))))
                visualize_wavefield(u_elapse.squeeze(), ut_elapse.squeeze(), vel=torch.from_numpy(vel), init_res_f=res, frame=True,
                                              f_delta_x=f_delta_x, f_delta_t=f_delta_t, it=s,
                                              optimization=optimization)

            if boundary_condition == "absorbing":
                # cropping and save current snapshot
                u_elapse, ut_elapse = WaveSol_from_EnergyComponent_tensor(u_n[0,0], u_n[0,1], u_n[0,2], vel, 2. / 128.,
                                                                          torch.sum(torch.sum(torch.sum(u_n[0,0]))))
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse, res), crop_center(ut_elapse, res)
                Ux[it, s], Uy[it, s], Utc[it, s] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)

            else:
                # save current snapshot
                Ux[it, s], Uy[it, s], Utc[it, s] = u_n[0,0], u_n[0,1], u_n[0,2]

            if s < n_snaps + 1:
                u_n = one_iteration_pseudo_spectral_tensor(torch.cat([u_n, torch.from_numpy(vel).unsqueeze(dim=0).unsqueeze(dim=0)], dim=1))

    np.savez(output_path+output_appendix+str(res)+"_"+optimization+".npz", vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":

    for index in range(1):
        generate_wave_from_medium(output_path = "../data/end_to_end_" + str(index),
                                  res = 256,
                                  boundary_condition = "periodic",
                                  visualize = True,
                                  index = str(index),
                                  n_it = 10,  # how many data samples to generate
                                  optimization = "parareal")



