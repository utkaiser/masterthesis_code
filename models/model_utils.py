from os import environ, path
from torch import load
import numpy as np
import torch

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"

def save_model(model, modelname):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    for i in range(100):
        saving_path = path.join(path.dirname(path.dirname(path.abspath(__file__))),'results/run_1/saved_model_' +modelname+ "_"+ str(i) + '.pt')
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


def fetch_data(data_paths, batchsize, shuffle=True):
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
                                                  batch_size=batchsize, shuffle=shuffle, num_workers=1)
        total_n_datapoints += len(data_loader)
        train_loaders.append(data_loader)

    print("total number of data points:", total_n_datapoints * batchsize)
    return train_loaders


def fetch_data_end_to_end(data_paths, batchsize, shuffle=True):
    print("setting up data")

    total_n_datapoints = 0
    train_loaders = []

    for path in data_paths:
        npz_PropS = np.load(path)
        inputdata = torch.stack((npdat2Tensor(npz_PropS['X_U']),
                                 npdat2Tensor(npz_PropS['X_Ut']),
                                 npdat2Tensor(npz_PropS['vel'])), dim=1)
        outputdata = torch.stack((npdat2Tensor(npz_PropS['Y_U']),
                                  npdat2Tensor(npz_PropS['Y_Ut'])), dim=1)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputdata, outputdata),
                                                  batch_size=batchsize, shuffle=shuffle, num_workers=1)
        total_n_datapoints += len(data_loader)
        train_loaders.append(data_loader)

    print("total number of data points:", total_n_datapoints * batchsize)
    return train_loaders