from os import environ
import os
import torch
from datetime import datetime

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_paths():
    '''
    Returns
    -------
    get relevant paths for end-to-end Parareal training;
    while we train our model, we perform Parareal iterations so that the model learns how to adjust the weights in this setting
    '''

    main_branch = '../results/run_2/'
    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)

    data_paths = [
        '../data/D_t_128_parareal/end_to_end_0diag__3l__cp__hf__bp_m256_parareal.npz',
        # '../data/D_t_128_parareal/end_to_end_1diag__3l__cp__hf__bp_m256_parareal.npz',
        # '../data/D_t_128_parareal/end_to_end_2diag__3l__cp__hf__bp_m256_parareal.npz',
        # '../data/D_t_128_parareal/end_to_end_3diag__3l__cp__hf__bp_m256_parareal.npz'
    ]
    val_paths = [
        '../../data/val/end_to_end_val_diag_128.npz',
        # '../data/val/end_to_end_val_bp_128.npz',
        # '../data/val/end_to_end_val_m_128.npz',
        # '../data/val/end_to_end_val_hf_128.npz',
        # '../data/val/end_to_end_val_3l_128.npz',
        # '../data/val/end_to_end_val_cp_128.npz'
    ]
    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths


def get_params(params="0"):
    '''
    Parameters
    ----------
    params : (string) description of parameter composition

    Returns
    -------
    get parameter for Parareal end-to-end training
    '''

    param_dict = {}

    if params == "0":
        param_dict["batch_size"] = 30
        param_dict["lr"] = .001
        param_dict["res_scaler"] = 2
        param_dict["n_epochs"] = 100
        param_dict["model_name"] = "end_to_end_only_unet3lvl"
        param_dict["model_res"] = 128
        param_dict["coarse_res"] = param_dict["model_res"] / param_dict["res_scaler"]
        param_dict["n_snaps"] = 11
        param_dict["flipping"] = False
        param_dict["boundary_c"] = "absorbing"
        param_dict["total_time"] = .6
        param_dict["delta_t_star"] = .06
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20
        param_dict["c_delta_x"] = 2./64.
        param_dict["c_delta_t"] = 1./ 600. #param_dict["c_delta_x"] / 12
        param_dict["n_epochs_save_model"] = 5
        param_dict["restriction_type"] = "interpolation"  # options: cnn, interpolation, simple
    else:
        raise NotImplementedError("params not defined for params =",params)

    return param_dict





