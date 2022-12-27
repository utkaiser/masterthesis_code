from os import environ, path
from torch import load
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
import scipy.stats as ss
from datetime import datetime
import logging

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"

def save_model(model, modelname, dir_path='results/run_2/'):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    for i in range(100):
        saving_path = dir_path + 'saved_model_' + modelname + "_"+ str(i) + '.pt'
        if not path.isfile(saving_path):
            return save(model.state_dict(), saving_path)
    raise MemoryError("memory exceeded")


def load_model(load_path, model):

    torch.load(load_path)
    return model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)),
                                                load_path), map_location='cpu'), strict=False)


def npdat2Tensor(nda):
    ndt = np.transpose(nda,(2,0,1))
    return torch.from_numpy(ndt)

def npdat2Tensor_tensor(nda):
    ndt = torch.permute(nda,(2,0,1))
    return ndt


def fetch_data(data_paths, batch_size=1, shuffle=True):
    logging.info("setting up data")

    total_n_datapoints = 0
    train_loaders = []

    for path in data_paths:
        npz_PropS = np.load(path)
        inputdata = torch.stack((npdat2Tensor(npz_PropS['Ucx']),
                                 npdat2Tensor(npz_PropS['Ucy']),
                                 npdat2Tensor(npz_PropS['Utc']),
                                 npdat2Tensor(npz_PropS['vel'])), dim=1)
        outputdata = torch.stack((npdat2Tensor(npz_PropS['Ufx']),
                                  npdat2Tensor(npz_PropS['Ufy']),
                                  npdat2Tensor(npz_PropS['Utf'])), dim=1)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputdata, outputdata),
                                                  batch_size=batch_size, shuffle=shuffle, num_workers=1)
        total_n_datapoints += len(data_loader)
        train_loaders.append(data_loader)

    logging.info(" ".join(["total number of data points:", total_n_datapoints * batch_size]))
    return train_loaders



def fetch_data_end_to_end(data_paths, batch_size, shuffle=True, train_split = .9, validate=False, val_paths = None):

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

    full_dataset = get_datasets(data_paths)
    val_dataset = get_datasets(val_paths)

    if val_paths != None:

        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

        logging.info(" ".join(["train data points:", str(len(train_loader) * batch_size), "| test data points:", str(len(val_loader) * batch_size)]))
        return train_loader, val_loader

    else:
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        logging.info(" ".join(["train data points:", len(train_loader) * batch_size, "| test data points:", len(val_loader) * batch_size]))
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

def sample_label_random(input_idx, label_distr_shift):
    possible_label_range = np.arange(input_idx + 2 - label_distr_shift, 12 - label_distr_shift)  # [a,b-1]
    prob = ss.norm.cdf(possible_label_range + 0.5, scale=3) - ss.norm.cdf(possible_label_range - 0.5, scale=3)
    label_range = list(np.random.choice(possible_label_range + label_distr_shift, size=1, p=prob / prob.sum()))[0]
    return label_range

import torch
from generate_data import wave_util
import matplotlib.pyplot as plt

def visualize_wavefield(tensor_list, dx = 2.0 / 128.0, f_delta_t=.06, scaler=2, vis_save=True, vis_path='../results/run_2'):
    # list of tupels with tensors

    n_snapshots = len(tensor_list)
    fig = plt.figure(figsize=(20, 8))
    loss_list = []
    f_delta_x = 2.0 / 128.0
    label_font_size = 5


    for i, values in enumerate(tensor_list):

        epoch, loss, input_idx, label_idx, input, output, label = values
        input, output, label = input.cpu(), output.cpu(), label.cpu()
        loss_list.append(str(round(loss,5)))

        combined_data = torch.stack([input[:3,:,:], output[:3,:,:], label])
        _min, _max = torch.min(combined_data), torch.max(combined_data)

        # velocity
        if i == 0:
            ax1 = fig.add_subplot(4, n_snapshots, i + 1)
            pos1 = ax1.imshow(input[3, :, :])
            plt.axis('off')
            c_v = plt.colorbar(pos1)
            c_v.ax.tick_params(labelsize=label_font_size)

        # input
        u_x, u_y, u_t_c, vel = input[0, :, :].unsqueeze(dim=0), input[1, :, :].unsqueeze(dim=0), input[2, :, :].unsqueeze(dim=0), input[3, :, :].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax2 = fig.add_subplot(4, n_snapshots, i+1 + n_snapshots)
        pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax2.set_title('input', fontsize=10)
        c_i = plt.colorbar(pos2)
        c_i.ax.tick_params(labelsize=label_font_size)

        # output
        u_x, u_y, u_t_c = output[0, :, :].unsqueeze(dim=0), output[1, :, :].unsqueeze(dim=0), output[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax3 = fig.add_subplot(4, n_snapshots, i + 1 +n_snapshots*2)
        pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax3.set_title('output', fontsize=10)
        c_o = plt.colorbar(pos3)
        c_o.ax.tick_params(labelsize=label_font_size)

        # label
        u_x, u_y, u_t_c = label[0, :, :].unsqueeze(dim=0), label[1, :, :].unsqueeze(dim=0), label[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_x, sumv)
        ax4 = fig.add_subplot(4, n_snapshots, i + 1 + n_snapshots*3)
        pos4 = ax4.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax4.set_title('label', fontsize=10)
        c_l = plt.colorbar(pos4)
        c_l.ax.tick_params(labelsize=label_font_size)

    fig.suptitle("losses: " + ", ".join(loss_list), fontsize=14)

    if vis_save:
        for i in range(500):
            saving_path = vis_path + "epoch_" + str(epoch) + "_img_" + str(i) + ".png"
            if not path.isfile(saving_path):
                plt.savefig(saving_path)
                break
    else:
        plt.show()

import os

def get_paths():

    main_branch = '../results/run_2/'
    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)

    data_paths = ['../data/end_to_end_bp_m_10_2000.npz']
    val_paths = ['../data/end_to_end_bp_m_20_diagonal_ray.npz']
                 #'../data/end_to_end_bp_m_10_2000.npz']
    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths

def get_params(params="0"):

    param_dict = {}

    if params == "0":
        param_dict["batch_size"] = 50
        param_dict["lr"] = .001
        param_dict["res_scaler"] = 2
        param_dict["n_epochs"] = 500
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
        param_dict["c_delta_t"] = 1./300. #param_dict["c_delta_x"] / 12
        param_dict["downsampling_net"] = False
    else:
        raise NotImplementedError("params not defined for params =",params)

    return param_dict


import torch.utils.tensorboard as tb
import time

def setup_logger(logging_bool, train_logger_path, valid_logger_path, model_name, model_res, vis_path):

    logging.basicConfig(filename=vis_path + "out.log",
                        filemode='a',
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


    train_logger, valid_logger = None, None
    if logging_bool:
        train_logger = tb.SummaryWriter(train_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
        valid_logger = tb.SummaryWriter(valid_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
    global_step = 0

    return train_logger, valid_logger, global_step


def min_max_scale(data, new_min = -1, new_max = 1):
    '''

    Parameters
    ----------
    data : tensor, b x n_snaps x 4 x w x h

    Returns
    -------
    tensor, b x n_snaps x 4 x w x h
    scaled data with values ranging from -1 to 1 image-wise
    '''

    batch_size, n_snaps, channels, w, h = data.shape
    new_data = torch.zeros(batch_size, n_snaps, channels, w, h)

    for b in range(batch_size):
        for s in range(n_snaps):
            for c in range(channels-1): #we do not rescale velocity

                var = data[b,s,c,:,:]

                if s == 0 and c == 2:
                    var_p = var
                else:
                    var_min, var_max = var.min(), var.max()
                    var_p = (var - var_min) / (var_max - var_min) * (new_max - new_min) + new_min

                new_data[b,s,c,:,:] = var_p
            new_data[b, s, 3, :, :] = data[b, s, 3, :, :]

    return new_data








