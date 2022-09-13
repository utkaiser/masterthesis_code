from os import environ, path
from torch import load
import numpy as np
import torch

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"

def save_model(model):
    from torch import save
    from os import path
    model.to(torch.device("cpu"))
    for i in range(20):
        saving_path = path.join(path.dirname(path.abspath(__file__)),'saved_model_' + str(i) + '.pt')
        if not path.isfile(saving_path):
            return save(model.state_dict(), saving_path)
    raise MemoryError("memory exceeded")


def load_model(load_path, model):
    torch.load('./'+load_path+'.pt')
    return model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)),
                                                load_path), map_location='cpu'))


def npdat2Tensor(nda):
    ndt = np.transpose(nda,(2,0,1))
    ndt = torch.from_numpy(ndt)
    return ndt