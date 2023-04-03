from os import environ, path
from scipy.stats import truncnorm
import random
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torchmetrics.functional import mean_squared_error as MSE
from torchmetrics.functional import mean_absolute_error as MAE
import torchvision.transforms.functional as TF
import logging
import time
import sys
from generate_data.utils import get_wavefield

sys.path.append("..")
environ["TOKENIZERS_PARALLELISM"] = "false"
environ["OMP_NUM_THREADS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")


def choose_loss_function(name):

    if name == "SmoothL1Loss":
        return torch.nn.SmoothL1Loss()
    elif name == "MSE":
        return torch.nn.MSELoss()
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






