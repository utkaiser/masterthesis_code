from os import environ, path
from torch import load
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
import scipy.stats as ss

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"

def save_model(model, modelname, dir_path='results/run_2/'):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    for i in range(100):
        saving_path = path.join(path.dirname(path.dirname(path.abspath(__file__))), dir_path + 'saved_model_' +modelname+ "_"+ str(i) + '.pt')
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
    print("setting up data")

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

    print("total number of data points:", total_n_datapoints * batch_size)
    return train_loaders


def fetch_data_end_to_end(data_paths, batch_size, shuffle=True, train_split = .9,validate=False):

    #concatenate
    for i, path in enumerate(data_paths):
        if i == 0: np_array = np.load(path) # 200 x 11 x 128 x 128
        else: np_array = np.concatenate((np_array, np.load(path)), axis=0)

    tensor = torch.stack((torch.from_numpy(np_array['Ux']),
                          torch.from_numpy(np_array['Uy']),
                          torch.from_numpy(np_array['Utc']),
                          torch.from_numpy(np_array['vel'])), dim=2)

    full_dataset = torch.utils.data.TensorDataset(tensor)

    if validate:

        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=shuffle, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        print("test data points:", len(train_loader) * batch_size, "| train data points:", len(val_loader) * batch_size)
        return train_loader, val_loader
    else:

        #trainset_1 = torch.utils.data.Subset(full_dataset, [0,1,2])
        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
        print("test data points:", len(train_loader) * batch_size)
        return train_loader, None

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
import numpy as np

def visualize_wavefield(tensor_list, dx = 2.0 / 128.0, f_delta_t=.06, scaler=2, save=True):
    # list of tupels with tensors

    n_snapshots = len(tensor_list)
    fig = plt.figure(figsize=(20, 8))
    loss_list = []

    for i, values in enumerate(tensor_list):

        loss, input_idx, label_idx, input, output, label = values
        loss_list.append(str(round(loss,5)))

        combined_data = torch.stack([input[:3,:,:], output[:3,:,:], label])
        _min, _max = torch.min(combined_data), torch.max(combined_data)

        # velocity
        if i == 0:
            ax1 = fig.add_subplot(4, n_snapshots, i + 1)
            pos1 = ax1.imshow(input[3, :, :])
            plt.axis('off')
            plt.colorbar(pos1)

        # input
        u_x, u_y, u_t_c, vel = input[0, :, :].unsqueeze(dim=0), input[1, :, :].unsqueeze(dim=0), input[2, :, :].unsqueeze(dim=0), input[3, :, :].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax2 = fig.add_subplot(4, n_snapshots, i+1 + n_snapshots)
        pos2 = ax2.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax2.set_title('input', fontsize=10)
        plt.colorbar(pos2)

        # output
        u_x, u_y, u_t_c = output[0, :, :].unsqueeze(dim=0), output[1, :, :].unsqueeze(dim=0), output[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax3 = fig.add_subplot(4, n_snapshots, i + 1 +n_snapshots*2)
        pos3 = ax3.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax3.set_title('output', fontsize=10)
        plt.colorbar(pos3)

        # label
        u_x, u_y, u_t_c = label[0, :, :].unsqueeze(dim=0), label[1, :, :].unsqueeze(dim=0), label[2, :,:].unsqueeze(dim=0)
        sumv = torch.sum(torch.sum(u_x))
        u, ut = wave_util.WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, sumv)
        ax4 = fig.add_subplot(4, n_snapshots, i + 1 + n_snapshots*3)
        pos4 = ax4.imshow(wave_util.WaveEnergyField_tensor(u[0, :, :], ut[0, :, :], vel[0, :, :], dx) * dx * dx)#,vmin = _min, vmax = _max)
        plt.axis('off')
        ax4.set_title('label', fontsize=10)
        plt.colorbar(pos4)

    fig.suptitle("losses: " + ", ".join(loss_list), fontsize=14)

    if save:
        plt.savefig('temp.png')
    else:
        plt.show()


def get_paths():

    data_paths = ['../data/end_to_end_bp_m_200_2000.npz']
    train_logger_path = '../results/run_2/log_train/'
    valid_logger_path = '../results/run_2/log_valid/'
    dir_path_save = 'results/run_2/'

    return data_paths, train_logger_path, valid_logger_path, dir_path_save

def get_params(params="0"):

    if params == "0":
        batch_size = 10
        lr = .001
        res_scaler = 2
        n_epochs = 500
        model_name = "end_to_end_unet3lvl"
        model_res = "128"
        flipping = False
        boundary_c = "absorbing"
        delta_t_star = .06
        f_delta_x = 2.0 / 128.0
    else:
        raise NotImplementedError("params not defined for params =",params)

    print("start training:", batch_size, lr, res_scaler, n_epochs, model_name, model_res, flipping, boundary_c, delta_t_star, f_delta_x)

    return batch_size, lr, res_scaler, n_epochs, model_name, model_res, flipping, boundary_c, delta_t_star, f_delta_x