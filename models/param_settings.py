import os
from datetime import datetime


def get_paths(
        config = 0
):
    '''
    Parameters
    ----------
    config : (int) configuration index of paths

    Returns
    -------
    (list) paths used for training including data paths and logger paths
    '''

    if config == 0:  # experiment 1
        data_paths = [

            # small for debugging
            '../data/Dt_128/bp_marmousi_128_10_none_14.npz',

            # 12.5 * 400 * 8 = 40,000
            '../data/Dt_128/bp_marmousi_128_400_none_0.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_1.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_2.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_3.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_4.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_5.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_6.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_7.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_8.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_9.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_10.npz',
            '../data/Dt_128/bp_marmousi_128_400_none_11.npz',
            '../data/Dt_128/bp_marmousi_128_200_none_12.npz',
        ]
        test_paths = [
            '../data/Dt_128/test_128_84_none_0.npz',
        ]

    elif config == 1:  # experiment 2
        data_paths = [
            '../data/Dt_128/bp_marmousi_384_10_none_14.npz'
        ]
        test_paths = [
            '../data/val/end_to_end_val_3l_384.npz',
        ]

    else:
        raise NotImplementedError("These path settings are not specified")

    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    main_branch_folder = "normal/"
    main_branch = '../results/run_5/' + main_branch_folder

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)
        os.makedirs(main_branch + add + "/validated_models")

    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, test_paths


def get_params(
        config = 0
):
    '''
    Parameters
    ----------
    mode : which parameter composition to use

    Returns
    -------
    (dictionary) get parameters used for training
    '''

    param_dict = {}

    if config == 0:
        param_dict["n_epochs"] = 2  # todo: change back
        param_dict["n_snaps"] = 8
        param_dict["boundary_c"] = "absorbing"
        param_dict["delta_t_star"] = .06
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.
        param_dict["c_delta_x"] = 2./64.
        param_dict["c_delta_t"] = 1./600.
        param_dict["n_epochs_save_model"] = 5
        param_dict["optimizer_name"] = "AdamW"
        param_dict["loss_function_name"] = "SmoothL1Loss"
        param_dict["res_scaler"] = 2

    else:
        raise NotImplementedError("params not defined for params =", config)

    return param_dict