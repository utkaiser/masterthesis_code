import logging
import os

import numpy as np
from torch import nn

from generate_data.utils_wave_propagate import one_iteration_velocity_verlet_tensor, resize_to_coarse
from models.model_end_to_end import setup_model, save_model
from models.model_upsampling import choose_upsampling
import torch
import random
from models.param_settings import get_params, get_paths
from models.utils import setup_logger, hyperparameter_grid_search_end_to_end, choose_optimizer, choose_loss_function
from models.visualize_training import visualize_wavefield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model_old_paper(
    nn.Module,
):
    def __init__(
            self,
            param_dict
    ):
        super().__init__()
        self.model_upsampling = choose_upsampling("UNet3", param_dict["res_scaler"])
        self.model_upsampling.to(device)



def train_Dt_old_paper(
        flipping,
        experiment_index,
        visualize_res_bool,
        vis_save,
        model_res = 128
):
    # params and logger setup
    data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths = get_paths(experiment_index)
    param_d = get_params(experiment_index, flipping)
    param_d['flipping'] = flipping
    model_name = f"{param_d['optimizer_name']}_{param_d['loss_function_name']}_{param_d['res_scaler']}_{flipping}"
    train_logger, valid_logger, global_step = setup_logger(False, train_logger_path, valid_logger_path,
                                                           model_name, model_res, vis_path)
    logging.info(" ".join(["data settings:\n", ",\n".join(data_paths)]))
    logging.info(" ".join(["param settings:\n", ",\n".join([f"{i}: {v}" for i, v in param_d.items()])]))
    logging.info(model_name)
    logging.info(f"gpu available: {torch.cuda.is_available()} | number of gpus: {torch.cuda.device_count()}")

    dict_best_score = {
        "score": float("inf"),
        "lr": None,
        "batch_size": None,
        "weight_decay": None,
    }

    logging.info(f"{'-' * 100}\n\nHyperparameter Search:")

    # training and hyperparameter search with validation
    for lr, batch_size, weight_decay, _, _ in hyperparameter_grid_search_end_to_end(experiment_index, param_d, "old_paper"):

        logging.info(
            f"{'-' * 10} Start training of {lr}, {batch_size}, {weight_decay}, {param_d['c_delta_x']:.5f}, {param_d['c_delta_t']:.5f} {'-' * 10}")

        model, optimizer, loss_f = setup_model_old_paper(param_d, lr, weight_decay)
        train_loader, val_loader, _ = fetch_data_old_paper(data_paths, batch_size, val_paths)

        for epoch in range(param_d["n_epochs"]):

            train_loss_list, model, global_step, optimizer = train_model(model, epoch, train_loader, param_d, global_step,optimizer, train_logger, loss_f)
            val_performance = val_model(model, train_logger, vis_path, vis_save, global_step,
                                        valid_logger, val_loader, loss_f, epoch, train_loss_list, visualize_res_bool)

            # for last iteration, check if performance is best so far
            if epoch + 1 == param_d["n_epochs"]:
                logging.info(
                    f"This iteration has a loss of {val_performance:.4f} compared to the best loss so far {dict_best_score['score']}")
                if dict_best_score["score"] > val_performance:
                    dict_best_score = {
                        "score": val_performance,
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                    }

        save_model(model,
                   f"{model_name}_{lr}_{batch_size}_{weight_decay}_{param_d['c_delta_x']:.5f}_{param_d['c_delta_t']:.5f}",
                   dir_path_save + "validated_models/")

    # testing
    logging.info(f"{'/' * 100}\n\nTesting of best parameters: {dict_best_score}")

    param_d["c_delta_x"] = dict_best_score["dx"]
    param_d["c_delta_t"] = dict_best_score["dt"]

    lr, batch_size, weight_decay = [v for v in dict_best_score.values()][1:]
    model, optimizer, loss_f = setup_model_old_paper(param_d, lr, weight_decay)
    train_loader, val_loader, test_loader = fetch_data_old_paper(data_paths, batch_size, val_paths)
    trainval_loader = [d for dl in [train_loader, val_loader] for d in dl]  # merge train and validation set for testing

    for epoch in range(param_d["n_epochs"]):
        train_loss_list, model, global_step, optimizer = train_model(model, epoch, train_loader, param_d, global_step, optimizer, train_logger, loss_f)
        _ = val_model(model, train_logger, vis_path, vis_save, global_step, valid_logger, test_loader,
                      loss_f, epoch, train_loss_list, visualize_res_bool)

    save_model(model, f"best_{model_name}_{lr}_{batch_size}_{weight_decay}", dir_path_save)


def train_model(model, epoch, train_loader, param_d, global_step, optimizer, train_logger, loss_f):
    model.train()
    train_loss_list = []

    for i, input_tensor, label_tensor in enumerate(train_loader):
        input_tensor, label_tensor = input_tensor[0].detach(), label_tensor[0].detach()

        for input_idx in random.choices(range(param_d["n_snaps"]+1), k=param_d["n_snaps"]):

            input_tensor = input_tensor[:, input_idx].detach()  # b x 4 x w x h
            label = label_tensor[:, input_idx]  # b x 3 x w x h
            output = model(input_tensor)  # b x 3 x w x h
            loss = loss_f(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss_list.append(loss)

    return train_loss_list, model, global_step, optimizer


def val_model(model, train_logger, vis_path, vis_save, global_step, valid_logger, val_loader, loss_f,
              epoch, train_loss_list, visualize_res_bool):
    model.eval()
    with torch.no_grad():

        val_loss_list = []
        for i, _, data in enumerate(val_loader):

            n_snaps = data[0].shape[1]
            data = data[0].to(device)  # b x n_snaps x 3 x w x h

            visualize_list = []
            input_tensor = data[:, 0].clone()  # b x 4 x w x h
            vel = input_tensor[:, 3].unsqueeze(dim=1)

            for label_idx in range(1, n_snaps):
                label = data[:, label_idx, :3]  # b x 3 x w x h

                input_tensor = resize_to_coarse(input_tensor, 64)  # reduce resolution
                input_tensor = one_iteration_velocity_verlet_tensor(input_tensor)
                output = model(input_tensor)
                val_loss = loss_f(output, label)
                val_loss_list.append(val_loss.item())

                if i == 0 and visualize_res_bool:
                    # save only first element of batch
                    visualize_list.append((val_loss.item(), output[0].detach().cpu(),
                                           label[0].detach().cpu()))
                input_tensor = torch.cat((output, vel), dim=1)

            if i == 0 and visualize_res_bool:
                visualize_wavefield(epoch, visualize_list, input_tensor[0, 3].cpu(), vis_save=vis_save,
                                    vis_path=vis_path, initial_u=data[0, 0].unsqueeze(dim=0).cpu())

        val_performance = np.array(val_loss_list).mean()

        if epoch % 1 == 0:
            logging.info(" ".join(['epoch %d, train loss: %.5f, val loss: %.5f' % (epoch + 1, np.array(train_loss_list).mean(), np.array(val_loss_list).mean())]))

    return val_performance




def setup_model_old_paper(param_d, lr, weight_decay):

    model = Model_old_paper(param_d).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    optimizer = choose_optimizer(param_d["optimizer_name"], model, lr, weight_decay)
    loss_f = choose_loss_function(param_d["loss_function_name"])

    return model, optimizer, loss_f


def fetch_data_old_paper(data_paths, batch_size, additional_test_paths):
    def get_datasets(data_paths):

        #concatenate paths
        datasets = []
        for _, path in enumerate(data_paths):
            np_array = np.load(path, allow_pickle=True)  # 200 x 11 x 128 x 128
            datasets.append(
                torch.utils.data.TensorDataset(
                    torch.stack((torch.from_numpy(np_array['Ux']),
                                 torch.from_numpy(np_array['Uy']),
                                 torch.from_numpy(np_array['Utc']),
                                 torch.from_numpy(np_array['vel'])
                                 ), dim=2),
                    torch.stack((torch.from_numpy(np_array['Ux_coarse']),
                                 torch.from_numpy(np_array['Uy_coarse']),
                                 torch.from_numpy(np_array['Utc_coarse']),
                                 torch.from_numpy(np_array['vel_coarse'])
                                 ), dim=2)
                    )
            )

        return torch.utils.data.ConcatDataset(datasets)


    # get full dataset
    full_dataset = get_datasets(data_paths)

    # get split sizes
    train_size = int(0.8 * len(full_dataset))
    val_or_test_size = int(0.1 * len(full_dataset))

    # split dataset randomly and append special validation/ test data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_or_test_size, val_or_test_size])
    val_datasets = val_dataset
    test_datasets = test_dataset + get_datasets(additional_test_paths)

    # get dataloader objects
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=1)

    logging.info(f"data points train: {len(train_loader) * batch_size} | val: {len(val_loader) * batch_size} | test: {len(test_loader) * batch_size}")
    return train_loader, val_loader, test_loader



from datetime import datetime

def get_paths(
        experiment_index = 0
):
    '''
    Parameters
    ----------
    experiment_index : (int) configuration index of paths

    Returns
    -------
    (list) paths used for training including data paths and logger paths
    '''

    # test experiment used for debugging
    if experiment_index == 0:

        data_paths = [ '../data_old/Dt_old_128/bp_marmousi_128_400_none_0.npz' ]

    # data paths for all experiments
    else:

        data_paths = [
            # 12.5 * 400 * 8 = 40,000
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_0.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_1.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_2.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_3.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_4.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_5.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_6.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_7.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_8.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_9.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_10.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_400_none_11.npz',
            '../data_old/Dt_old_128/bp_marmousi_128_200_none_12.npz',
        ]

    test_paths = [
        '../data_old/Dt_old_128/test_128_84_none_0.npz',
    ]

    now = datetime.now()
    add = now.strftime("%d_%m_%Y____%H_%M_%S") + "/"

    main_branch_folder = f"exp{experiment_index}/"
    main_branch = '../results_old_paper/run_5/' + main_branch_folder

    if not os.path.exists(main_branch + add):
        os.makedirs(main_branch + add)
        os.makedirs(main_branch + add + "/validated_models")

    train_logger_path = main_branch + add + 'log_train/'
    valid_logger_path = main_branch + add + 'log_valid/'
    dir_path_save = main_branch + add
    vis_path = main_branch + add

    return data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, test_paths







# todo: test data