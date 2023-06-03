import os
from datetime import datetime


def get_paths(
        model_res
):
    '''
    Parameters
    ----------
    model_res : (int) resolution of model

    Returns
    -------
    (list) paths used for training including data paths and logger paths
    '''

    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    if model_res == 128: data_suffix = "/D_t_128"
    else: data_suffix = "/D_t_256"
    main_branch_folder = ""
    main_branch = '../results/run_4/' + main_branch_folder

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)
    data_paths = [
        '../data/D_t_128/end_to_end_0diag__3l__cp__hf__bp_m128_none.npz'
        # '../data' + data_suffix + '/end_to_end_1diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        # '../data' + data_suffix + '/end_to_end_2diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        # '../data' + data_suffix + '/end_to_end_3diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        # '../data' + data_suffix + '/end_to_end_0diag__3l__cp__hf__bp_m' + str(model_res) + '.npz'
        # '../data' + data_suffix + '/end_to_end_4diag__3l__cp__hf__bp_m' + str(model_res) + '.npz'
    ]
    val_paths = [
        '../data/val/end_to_end_val_3l_128.npz',
        # '../data/val/end_to_end_val_bp_128.npz',
        # '../data/val/end_to_end_val_cp_128.npz',
        # '../data/val/end_to_end_val_diag_128.npz',
        # '../data/val/end_to_end_val_hf_128.npz',
        # '../data/val/end_to_end_val_m_128.npz'
    ]
    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths


def get_params(
        mode="0"
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

    if mode == "0":
        param_dict["batch_size"] = 1
        param_dict["lr"] = .001
        param_dict["n_epochs"] = 30
        param_dict["n_snaps"] = 8
        param_dict["flipping"] = False
        param_dict["boundary_c"] = "absorbing"
        param_dict["delta_t_star"] = .06
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.
        param_dict["c_delta_x"] = 2./64.
        param_dict["c_delta_t"] = 1./600.
        param_dict["n_epochs_save_model"] = 5
        param_dict["restriction_type"] = "interpolation"
        param_dict["res_scaler"] = 2

    else:
        raise NotImplementedError("params not defined for params =",params)

    return param_dict