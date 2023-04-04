import sys
sys.path.append("..")
import torch
import logging
import numpy as np
from initial_conditions import initial_condition_gaussian
from utils import crop_center, start_logger_datagen_end_to_end, get_resolution_padding
from change_wave_arguments import WaveEnergyComponentField_end_to_end, WaveSol_from_EnergyComponent_tensor
from generate_data.visualize_datagen import visualize_wavefield
from generate_velocity_profiles import get_velocities
from generate_data.param_settings import get_training_params
from generate_data.utils_wave_propagate import one_iteration_pseudo_spectral_tensor


def generate_wave_from_medium(
        output_dir,
        res,
        boundary_condition,
        visualize,
        index,
        n_it,
        optimization,
        velocity_profiles
):
    '''
    Parameters
    ----------
    output_dir : (string) director to save output in
    res : (int) resolution of input velocity profiles
    boundary_condition : (string) defines the boundary condition used for pde
    visualize : (bool) visualize generated data points
    index : (int) dataset number, used when generated multiple instances of same dataset type
    n_it : (int) number of different velocity profiles and wave advancement groups (see paper definition),
                    how many data samples to generate
    optimization : (string) optimization technique; "parareal" or "none"
    velocity_profiles : (string) type of velocity profiles "bp_marmousi" or "mixed" (all velocity profiles implemented)

    Returns
    -------
    - generates data to use for training the end-to-end model and stores it in a `.npz`-file;
    - for each velocity profiles, the fine solver (RK4-method) is used to advance a wave n_snaps times;
    - the parameters are defined in `param_settings.py`,
    - and we can choose between absorbing and periodic boundary conditions
    - data is stored as in energy components as we get training of neural networks
    '''

    # parameter setup
    res_padded = get_resolution_padding(boundary_condition, res, optimization)
    output_path = output_dir + velocity_profiles + "_" + str(res_padded) + "_" + str(n_it) + "_" \
                  + str(optimization) + "_" + boundary_condition + "_" + str(index)
    start_logger_datagen_end_to_end(output_path)
    param_dict = get_training_params(res)
    total_time, delta_t_star, f_delta_x, f_delta_t, n_snaps = \
        param_dict["total_time"], param_dict["delta_t_star"], \
        param_dict["f_delta_x"], param_dict["f_delta_t"], param_dict["n_snaps"]

    # data setup
    velocities = get_velocities(n_it, res_padded, velocity_profiles, optimization, boundary_condition)

    # tensors setup
    Ux, Uy, Utc = np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res]), np.zeros([n_it, n_snaps + 1, res, res])
    V = np.zeros([n_it, n_snaps+1, res, res])

    # training
    logging.info(" ".join(["start end to end training data generation, amount of data to generate:", str(n_it)]))

    for it in range(velocities.shape[0]):
        logging.info(" ".join(['sample:', str(it)]))

        #initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_n = initial_condition_gaussian(torch.from_numpy(vel), res, boundary_condition, optimization,
                                         "energy_components", res_padded)  # 1 x 3 x 256 x 256

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
                u_elapse, ut_elapse = WaveSol_from_EnergyComponent_tensor(u_n[:,0], u_n[:,1], u_n[:,2], torch.from_numpy(vel), 2. / 128.,
                                                                          torch.sum(torch.sum(torch.sum(u_n[:,0]))))
                u_elapse_crop, ut_elapse_crop = crop_center(u_elapse[0], res), crop_center(ut_elapse[0], res)
                Ux[it, s], Uy[it, s], Utc[it, s] = WaveEnergyComponentField_end_to_end(u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x)

            else:
                # save current snapshot
                Ux[it, s], Uy[it, s], Utc[it, s] = u_n[0,0], u_n[0,1], u_n[0,2]

            if s < n_snaps + 1:
                u_n = one_iteration_pseudo_spectral_tensor(torch.cat([u_n, torch.from_numpy(vel).unsqueeze(dim=0).unsqueeze(dim=0)], dim=1))

    np.savez(f"{output_path}.npz", vel=V, Ux=Ux, Uy=Uy, Utc=Utc)


if __name__ == "__main__":

    for index in range(1):  # run multiple iterations of datagen
        generate_wave_from_medium(
            output_dir = "../data/",
            visualize = True,
            n_it = 200,
            res=128,
            boundary_condition="absorbing",
            optimization = "parareal",
            index = index,
            velocity_profiles = "bp_marmousi"
        )



