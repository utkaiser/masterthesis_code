import os
from datetime import datetime

import sys
sys.path.append("..")
sys.path.append("../..")


def get_paths(experiment_index=0):
    """
    Parameters
    ----------
    experiment_index : (int) configuration index of paths

    Returns
    -------
    (list) paths used for training including data paths and logger paths
    """

    # test experiment used for debugging
    if experiment_index == 0:  # todo: change back
        data_paths = ["../data/Dt_128/bp_marmousi_128_10_none_14.npz"]

    # data paths for all experiments
    else:
        data_paths = [
            # 12.5 * 400 * 8 = 40,000
            "../data/Dt_128/bp_marmousi_128_400_none_0.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_1.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_2.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_3.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_4.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_5.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_6.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_7.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_8.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_9.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_10.npz",
            "../data/Dt_128/bp_marmousi_128_400_none_11.npz",
            "../data/Dt_128/bp_marmousi_128_200_none_12.npz",
        ]

    test_paths = [
        "../data/Dt_128/test_128_84_none_0.npz",
    ]

    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    main_branch_folder = f"exp{experiment_index}/"
    main_branch = "../results/run_5/" + main_branch_folder

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)
        os.makedirs(main_branch + add + "/validated_models")
        os.makedirs(main_branch + add + "vis_results")

    train_logger_path = main_branch + add + "log_train/"
    valid_logger_path = main_branch + add + "log_valid/"
    dir_path_save = main_branch + add
    vis_path = main_branch + add + "vis_results/"

    return (
        data_paths,
        train_logger_path,
        valid_logger_path,
        dir_path_save,
        vis_path,
        test_paths,
    )


def get_params(experiment_index, flipping):
    """
    Parameters
    ----------
    experiment_index : (int) which parameter composition to use

    Returns
    -------
    (dictionary) get parameters used for training
    """

    d = {
        "n_epochs": 20,  # todo: change back
        "n_snaps": 8,
        "boundary_c": "absorbing",
        "delta_t_star": 0.06,
        "f_delta_x": 2.0 / 128.0,
        "f_delta_t": (2.0 / 128.0) / 20.0,
        "c_delta_x": 2.0 / 64.0,
        "c_delta_t": 1.0 / 600.0,
        "optimizer_name": "AdamW",
        "loss_function_name": "MSE",
        "res_scaler": 2,
    }

    if experiment_index == 12:  # unet interpolation and longer training
        d["n_epochs"] *= 4

    return d
