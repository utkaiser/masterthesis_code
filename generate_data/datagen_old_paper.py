import sys

sys.path.append("..")
sys.path.append("../..")
import logging

import numpy as np
import torch
from change_wave_arguments import (
    WaveEnergyComponentField_end_to_end,
    WaveSol_from_EnergyComponent_tensor,
)
from generate_velocity_profiles import get_velocities
from initial_conditions import initial_condition_gaussian
from utils import (
    crop_center,
    crop_center_tensor,
    get_resolution_padding,
    start_logger_datagen_end_to_end,
)

from generate_data.param_settings import get_training_params
from generate_data.utils_wave_propagate import (
    one_iteration_pseudo_spectral_tensor,
    one_iteration_velocity_verlet_tensor,
    resize_to_coarse_interp,
)
from generate_data.visualize_datagen import (
    visualize_wavefield,
    visualize_wavefield_old_paper,
)


def generate_wave_from_medium(
    output_dir, res, visualize, index, n_it, optimization_technique, velocity_profiles
):
    """
    Parameters
    ----------
    output_dir : (string) director to save output in
    res : (int) resolution of input velocity profiles
    visualize : (bool) visualize generated data points
    index : (int) dataset number, used when generated multiple instances of same dataset type
    n_it : (int) number of different velocity profiles and wave advancement groups (see paper definition),
                    how many data samples to generate
    optimization_technique : (string) optimization technique; "parareal" or "none"
    velocity_profiles : (string) type of velocity profiles "bp_marmousi" or "mixed" (all velocity profiles implemented)

    Returns
    -------
    - generates data to use for training the end-to-end model and stores it in a `.npz`-file;
    - for each velocity profiles, the fine solver (RK4-method) is used to advance a wave n_snaps times;
    - the parameters are defined in `param_settings.py`,
    - data is stored as in energy components as we get training of neural networks
    """

    # parameter setup
    res_padded = get_resolution_padding(res, optimization_technique)
    output_path = f"{output_dir}/Dt_old_{res}/{velocity_profiles}_{res_padded}_{n_it}_{optimization_technique}_{index}"
    start_logger_datagen_end_to_end(output_path)
    param_dict = get_training_params(res)
    delta_t_star, f_delta_x, f_delta_t, n_snaps = (
        param_dict["delta_t_star"],
        param_dict["f_delta_x"],
        param_dict["f_delta_t"],
        param_dict["n_snaps"],
    )
    c_delta_x, c_delta_t = param_dict["c_delta_x"], param_dict["c_delta_t"]
    res_coarse = res // 2

    # data setup
    velocities = get_velocities(res_padded, velocity_profiles, optimization_technique)
    np.random.shuffle(velocities)

    # tensors setup
    Ux, Uy, Utc = (
        np.zeros([n_it, n_snaps + 1, res, res]),
        np.zeros([n_it, n_snaps + 1, res, res]),
        np.zeros([n_it, n_snaps + 1, res, res]),
    )
    Ux_coarse, Uy_coarse, Utc_coarse = (
        np.zeros([n_it, n_snaps + 1, res_coarse, res_coarse]),
        np.zeros([n_it, n_snaps + 1, res_coarse, res_coarse]),
        np.zeros([n_it, n_snaps + 1, res_coarse, res_coarse]),
    )
    V = np.zeros([n_it, n_snaps + 1, res, res])
    V_coarse = np.zeros([n_it, n_snaps + 1, res_coarse, res_coarse])

    # training
    logging.info(
        f"start end to end training data generation, amount of data to generate: {n_it}"
    )

    for it in range(n_it):
        logging.info(f"sample: {it}")

        # initialization of wave field
        vel = velocities[it]  # w_big x h_big
        u_n = initial_condition_gaussian(
            torch.from_numpy(vel),
            res,
            optimization_technique,
            "energy_components",
            res_padded,
        )
        u_n_coarse = resize_to_coarse_interp(u_n, res)

        # create and save velocity crop
        vel_crop = crop_center(vel, res, 2)
        V[it] = np.repeat(vel_crop[np.newaxis], n_snaps + 1, axis=0)
        vel_crop_coarse = crop_center(vel_crop, res_coarse)
        V_coarse[it] = np.repeat(vel_crop_coarse[np.newaxis], n_snaps + 1, axis=0)

        for s in range(n_snaps + 1):
            # integrate delta_t_star step size n_snaps times

            if visualize:
                visualize_wavefield_old_paper(
                    crop_center_tensor(u_n_coarse, res_coarse),
                    crop_center_tensor(u_n, res),
                    c_delta_x,
                    c_delta_t,
                    f_delta_x,
                    f_delta_t,
                    torch.from_numpy(vel),
                    torch.from_numpy(vel_crop),
                    torch.from_numpy(vel_crop_coarse),
                    s,
                )

            # cropping and save current snapshot
            u_elapse, ut_elapse = WaveSol_from_EnergyComponent_tensor(
                u_n[:, 0],
                u_n[:, 1],
                u_n[:, 2],
                torch.from_numpy(vel),
                f_delta_x,
                torch.sum(torch.sum(torch.sum(u_n[:, 0]))),
            )
            u_elapse_crop, ut_elapse_crop = crop_center(u_elapse[0], res), crop_center(
                ut_elapse[0], res
            )
            Ux[it, s], Uy[it, s], Utc[it, s] = WaveEnergyComponentField_end_to_end(
                u_elapse_crop, ut_elapse_crop, vel_crop, f_delta_x
            )

            u_elapse_coarse, ut_elapse_coarse = WaveSol_from_EnergyComponent_tensor(
                u_n_coarse[:, 0],
                u_n_coarse[:, 1],
                u_n_coarse[:, 2],
                torch.from_numpy(vel_crop),
                c_delta_x,
                torch.sum(torch.sum(torch.sum(u_n_coarse[:, 0]))),
            )
            u_elapse_crop_coarse, ut_elapse_crop_coarse = crop_center(
                u_elapse_coarse[0], res_coarse
            ), crop_center(ut_elapse_coarse[0], res_coarse)
            (
                Ux_coarse[it, s],
                Uy_coarse[it, s],
                Utc_coarse[it, s],
            ) = WaveEnergyComponentField_end_to_end(
                u_elapse_crop_coarse, ut_elapse_crop_coarse, vel_crop_coarse, c_delta_x
            )

            if s < n_snaps + 1:
                u_n_coarse = one_iteration_velocity_verlet_tensor(
                    torch.cat(
                        [
                            resize_to_coarse_interp(u_n, res),
                            torch.from_numpy(vel_crop)
                            .unsqueeze(dim=0)
                            .unsqueeze(dim=0),
                        ],
                        dim=1,
                    ),
                    c_delta_x,
                    c_delta_t,
                )
                u_n = one_iteration_pseudo_spectral_tensor(
                    torch.cat(
                        [u_n, torch.from_numpy(vel).unsqueeze(dim=0).unsqueeze(dim=0)],
                        dim=1,
                    ),
                    f_delta_x,
                    f_delta_t,
                )

    np.savez(
        f"{output_path}.npz",
        vel=V,
        Ux=Ux,
        Uy=Uy,
        Utc=Utc,
        vel_coarse=V_coarse,
        Ux_coarse=Ux_coarse,
        Uy_coarse=Uy_coarse,
        Utc_coarse=Utc_coarse,
    )


if __name__ == "__main__":
    for index in range(0, 13):  # run multiple iterations of datagen
        generate_wave_from_medium(
            output_dir="../data_old",
            visualize=False,
            n_it=400,
            res=128,
            optimization_technique="none",
            index=index,
            velocity_profiles="bp_marmousi",  # mixed
        )
