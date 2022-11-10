from os import environ, path
from torch import load
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random
import scipy.stats as ss

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"

def save_model(model, modelname):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    for i in range(100):
        saving_path = path.join(path.dirname(path.dirname(path.abspath(__file__))),'results/run_2/saved_model_' +modelname+ "_"+ str(i) + '.pt')
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


def fetch_data_end_to_end(data_paths, batch_size, shuffle=True, train_split = .9):
    global np_array
    print("setting up data")

    #concatenate
    for i, path in enumerate(data_paths):
        if i == 0: np_array = np.load(path) # 200 x 11 x 128 x 128
        else: np_array = np.concatenate((np_array, np.load(path)), axis=0)

    tensor = torch.stack((torch.from_numpy(np_array['Ux']),
                          torch.from_numpy(np_array['Uy']),
                          torch.from_numpy(np_array['Utc']),
                          torch.from_numpy(np_array['vel'])), dim=2)

    full_dataset = torch.utils.data.TensorDataset(tensor)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    #train_dataset = torch.utils.data.Subset(full_dataset, [0])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=1)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    print("test data points:", len(train_loader) * batch_size, "| train data points:", len(val_loader) * batch_size)
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