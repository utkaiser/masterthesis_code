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


def fetch_data_end_to_end(
        data_paths,
        batch_size,
        additional_test_paths
):
    '''
    Parameters
    ----------
    data_paths : (string) data paths to use for training and validation
    batch_size : (int) batch size
    additional_test_paths : (string) data paths to use for testing

    Returns
    -------
    return torch.Dataloader object to iterate over training, validation and testing samples
    '''

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


    # get full dataset
    full_dataset = get_datasets(data_paths)

    # get split sizes
    train_size = int(0.8 * len(full_dataset))
    val_or_test_size = int(0.1 * len(full_dataset))

    # split dataset randomly and append special validation/ test data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_or_test_size, val_or_test_size])
    val_datasets = val_dataset  # + get_datasets(additional_test_paths)
    test_datasets = test_dataset + get_datasets(additional_test_paths)

    # get dataloader objects
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=1)

    logging.info(f"data points train: {len(train_loader) * batch_size} | val: {len(val_loader) * batch_size} | test: {len(test_loader) * batch_size}")
    return train_loader, val_loader, test_loader


def flip_tensors(
        input_tensor,
        label,
        v_flipped,
        h_flipped
):
    '''
    Parameters
    ----------
    input_tensor : (pytorch tensor) wave img representation
    label : (pytorch tensor) label wave img representation
    v_flipped : (bool) if input_tensor should be vertically flipped
    h_flipped : (bool) if input_tensor should be horizontally flipped

    Returns
    -------
    flipped input and label according to flipping scheme with flipping probability of .5
    '''

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


def sample_label_normal_dist(
        input_idx,
        n_snaps,
        label_distr_shift,
        multi_step,
        weighted_loss
):
    '''
    Parameters
    ----------
    input_idx : (int) input index; i.e. which snapshot / wave advancement we consider currently
    n_snaps : (int) total number of snapshots / wave advancement for this velocity profile and initial condition
    label_distr_shift : (int) difference between current snapshot and how many wave advancement we would like
    multi_step : (bool) decides if multi-step loss is used (see paper)

    Returns
    -------
    randomly sample label idx from normal distribution
    used especially for multi-step approach, single-step approach also usable in this function
    '''

    if multi_step:

        if weighted_loss:

            low = input_idx
            upp = n_snaps
            mean = min(upp, input_idx + label_distr_shift)
            sd = 1
            if input_idx + 1 == n_snaps - 1:
                return n_snaps - 1
            else:
                return round(truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs())

        else:

            return random.randint(min(input_idx + 2, n_snaps), n_snaps)

    else:
        return min(input_idx + 2, n_snaps)


def setup_logger(
        logging_bool,
        train_logger_path,
        valid_logger_path,
        model_name,
        model_res,
        vis_path
):
    '''
    Parameters
    ----------
    logging_bool : (bool) decides if results are logged
    train_logger_path : (string) path used to save logs training
    valid_logger_path : (string) path used to save logs validation
    model_name : (string) name of model
    model_res : (int) resolution model can handle
    vis_path : (string) path used to save visualization

    Returns
    -------
    multiple loggers set up for our framework
    '''

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


def choose_optimizer(
        name,
        model,
        lr,
        weight_decay
):
    '''
    Parameters
    ----------
    name : (string) name of optimizer
    model : (pytorch.model) end-to-end model instance
    lr : (float) learning rate
    weight_decay: (float) weight decay

    Returns
    -------

    '''

    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer not implemented.")


def choose_loss_function(
        name
):
    '''
    Parameters
    ----------
    name : name of loss function

    Returns
    -------
    get pytorch loss function
    '''

    if name == "SmoothL1Loss":
        return torch.nn.SmoothL1Loss()
    elif name == "MSE":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError("Loss function not implemented.")


def relative_frobenius_norm(
        a,
        b
):
    '''
    Parameters
    ----------
    a : (pytorch tensor) matrix
    b : (pytorch tensor) matrix

    Returns
    -------
    compute relative frobenius norm for matrix a and b
    '''
    diff = a - b
    norm_diff = torch.sqrt(torch.sum(diff**2))
    norm_b = torch.sqrt(torch.sum(b**2))
    return norm_diff / norm_b


def compute_loss2(
        prediction,
        target
):
    '''
    Parameters
    ----------
    prediction : (pytorch tensor) prediction matrix
    target : (pytorch tensor) target matrix

    Returns
    -------
    get mean squared error loss between {prediction} and {target}
    '''
    return MSE(prediction, target).item()


def compute_loss(
        prediction,
        target,
        vel,
        mode = "MSE/frob(fine)"
):
    '''
    Parameters
    ----------
    prediction : (pytorch tensor) prediction matrix
    target : (pytorch tensor) target matrix
    vel : (pytorch tensor) velocity profile
    mode : type of error function

    Returns
    -------
    get error computed depending on choice of {mode}
    '''

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


def round_loss(
        number
):
    '''
    Parameters
    ----------
    number : (float) number to round

    Returns
    -------
    get rounding of {number}
    '''

    return number #str(round(number*(10**7),5))+"e-7"




def hyperparameter_grid_search_end_to_end(
        experiment_index,
        param_d,
        suffix = "end_to_end"
):
    '''
    experiment_index: (int) number of the experiment, explained in paper
    param_d: (dict) contains all relevant parameter settings

    Returns
    -------
    performs grid search to run end-to-end model on different hyperparameters
    '''

    list_lr = [
        .001,
        .0001
    ]

    list_batch_size = [
        2**6,
        2**8
    ]

    list_weight_decay = [
        .01,
        .001
    ]

    list_c_delta_x = [
        2 ** -6,
        2 ** -5,
        2 ** -4
    ]

    list_c_delta_t = [
        2 ** -10,
        2 ** -9,
        2 ** -8
    ]

    list_all_allocations = []

    if experiment_index == 0:
        return [(.001,2**6,.01,param_d["c_delta_x"],param_d["c_delta_t"])]

    else:
        for lr in list_lr:
            for bs in list_batch_size:
                for wd in list_weight_decay:

                    if experiment_index != 2 or suffix == "old_paper":
                        list_all_allocations.append((lr, bs, wd, param_d["c_delta_x"],param_d["c_delta_t"]))
                    else:
                        # EXPERIMENT 2
                        for dx in list_c_delta_x:
                            for dt in list_c_delta_t:
                                list_all_allocations.append((lr, bs, wd, dx, dt))

        return list_all_allocations


def vanilla_benchmark_grid_search_end_to_end(
        param_d,
        experiment_index

):

    list_all_allocations = []

    if experiment_index == 2:
        list_c_delta_x = [
            2 ** -6,
            2 ** -5,
            2 ** -4
        ]

        list_c_delta_t = [
            2 ** -10,
            2 ** -9,
            2 ** -8
        ]

        for dx in list_c_delta_x:
            for dt in list_c_delta_t:
                list_all_allocations.append((dx, dt))

    else:

        list_all_allocations.append((param_d["c_delta_x"], param_d["c_delta_t"]))

    return list_all_allocations

