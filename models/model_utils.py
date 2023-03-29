from os import environ, path
from scipy.stats import truncnorm
from torch import load
import os
import torchvision.transforms.functional as TF
import random
import numpy as np
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor, WaveSol_from_EnergyComponent, WaveEnergyField
from torchmetrics.functional import mean_squared_error as MSE
from torchmetrics.functional import mean_absolute_error as MAE
from datetime import datetime
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tb
import time
import sys
sys.path.append("..")


environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, modelname, dir_path='results/run_3/'):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    saving_path = dir_path + 'saved_model_' + modelname + '.pt'
    if not path.isfile(saving_path):
        return save(model.state_dict(), saving_path)
    else:
        raise MemoryError("File (.pt) already exists.")


def load_model(load_path, model):

    torch.load(load_path)
    return model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)),
                                                load_path), map_location='cpu'), strict=False)


def fetch_data_end_to_end(data_paths, batch_size, val_paths):

    def get_datasets(data_paths):

        #concatenate paths
        datasets = []
        for i, path in enumerate(data_paths):
            np_array = np.load(path)  # 200 x 11 x 128 x 128
            datasets.append(
                torch.utils.data.TensorDataset(
                    torch.stack((torch.from_numpy(np_array['Ux']),
                                 torch.from_numpy(np_array['Uy']),
                                 torch.from_numpy(np_array['Utc']),
                                 torch.from_numpy(np_array['vel'])), dim=2)
            ))
        return torch.utils.data.ConcatDataset(datasets)

    val_dataset = get_datasets(val_paths)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    full_dataset = get_datasets(data_paths)
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    logging.info(" ".join(["train data points:", str(len(train_loader) * batch_size), "| test data points:", str(len(val_loader) * batch_size)]))
    return train_loader, val_loader


def flip_tensors(input_tensor, label, v_flipped, h_flipped):

    if v_flipped: label = TF.vflip(label)
    if h_flipped: label = TF.hflip(label)

    #random vertical and horizontal flipping
    if random.random() > 0.5:
        v_flipped = not v_flipped
        input_tensor = TF.vflip(input_tensor)
        label = TF.vflip(label)
    if random.random() > 0.5:
        h_flipped = not h_flipped
        input_tensor = TF.hflip(input_tensor)
        label = TF.hflip(label)

    return input_tensor, label, v_flipped, h_flipped


def sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, multi_step):
    # randomly sample label idx from normal distribution
    if multi_step == 1:
        return input_idx + 1
    elif multi_step == 2:
        return min(n_snaps-1, input_idx + 2)
    else:  # multi_step == -1; therefore, shifting normal distribution
        low = input_idx + 1
        upp = n_snaps - 1
        mean = min(upp, input_idx + label_distr_shift)
        sd = 1
        if input_idx + 1 == n_snaps - 1:
            return n_snaps - 1
        else:
            return round(truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs())


def get_paths(model_res):

    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    if model_res == 128: data_suffix = "/D_t_128"
    else: data_suffix = "/D_t_256"
    val_res_factor = 1
    main_branch_folder = ""
    main_branch = '../results/run_3/' + main_branch_folder

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)
    data_paths = [
        # '../data/D_t_128/end_to_end_0diag__3l__cp__hf__bp_m128_none.npz'
        '../data' + data_suffix + '/end_to_end_1diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        '../data' + data_suffix + '/end_to_end_2diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        '../data' + data_suffix + '/end_to_end_3diag__3l__cp__hf__bp_m' + str(model_res) + '.npz',
        '../data' + data_suffix + '/end_to_end_0diag__3l__cp__hf__bp_m' + str(model_res) + '.npz'
        '../data' + data_suffix + '/end_to_end_4diag__3l__cp__hf__bp_m' + str(model_res) + '.npz'
    ]
    val_paths = [
        '../data/val/end_to_end_val_3l_128.npz',
        '../data/val/end_to_end_val_bp_128.npz',
        '../data/val/end_to_end_val_cp_128.npz',
        '../data/val/end_to_end_val_diag_128.npz',
        '../data/val/end_to_end_val_hf_128.npz',
        '../data/val/end_to_end_val_m_128.npz'
    ]
    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths


def get_params(params="0"):

    param_dict = {}

    if params == "0":
        param_dict["batch_size"] = 1
        param_dict["lr"] = .001
        param_dict["n_epochs"] = 20
        param_dict["n_snaps"] = 7
        param_dict["flipping"] = False
        param_dict["boundary_c"] = "absorbing"
        param_dict["total_time"] = .6
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


def setup_logger(logging_bool, train_logger_path, valid_logger_path, model_name, model_res, vis_path):

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(vis_path + '.log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    train_logger, valid_logger = None, None
    if logging_bool:
        train_logger = tb.SummaryWriter(train_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
        valid_logger = tb.SummaryWriter(valid_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
    global_step = 0

    return train_logger, valid_logger, global_step


def choose_optimizer(name, model, lr):

    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr)
    elif name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")


def choose_loss_function(name):

    if name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    elif name == "MSE":
        return nn.MSELoss()
    else:
        raise NotImplementedError("Loss function not implemented.")


def relative_frobenius_norm(a, b):
    diff = a - b
    norm_diff = torch.sqrt(torch.sum(diff**2))
    norm_b = torch.sqrt(torch.sum(b**2))
    return norm_diff / norm_b


def compute_loss2(prediction, target):
    return MSE(prediction, target).item()


def compute_loss(prediction,target, vel, mode = "MSE/frob(fine)"):

    prediction, target = get_wavefield(prediction, vel), get_wavefield(target,vel)

    if mode=="relative_frobenius":
        return relative_frobenius_norm(prediction, target).item()
    elif mode == "MSE":
        return MSE(prediction, target).item()
    elif mode == "MAE":
        return MAE(prediction, target).item()
    elif mode == "MSE/frob(fine)":
        return MSE(prediction, target).item() / torch.linalg.norm(target).item()
    elif mode == "MAE/frob(fine)":
        return MAE(prediction, target).item() / torch.linalg.norm(target).item()
    else:
        raise NotImplementedError("This mode has not been implemented yet.")


def round_loss(number):
    return number #str(round(number*(10**7),5))+"e-7"


def smaller_crop(matrix):
    # matrix -> b? x c x w x h
    if matrix.shape[-1] == 256:
        v = 64
    else:
        v = 32

    if len(matrix.shape) == 3:
        return matrix[:, v:-v, v:-v]
    elif len(matrix.shape) == 4:
        return matrix[:,:,v:-v, v:-v]
    elif len(matrix.shape) == 5:
        return matrix[:,:,:,v:-v, v:-v]
    elif len(matrix.shape) == 6:
        return matrix[:,:,:,:,v:-v, v:-v]

    else:
        return matrix[v:-v, v:-v]


def get_wavefield(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t,
                                                 torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                  f_delta_x) * f_delta_x * f_delta_x


def get_wavefield_numpy(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :]
    u, u_t = WaveSol_from_EnergyComponent(u_x, u_y, u_t_c, vel, f_delta_t,
                                                 np.sum(np.sum(np.sum(u_x))))
    return WaveEnergyField(u, u_t, vel,
                                  f_delta_x) * f_delta_x * f_delta_x






