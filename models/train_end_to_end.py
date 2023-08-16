import sys

import os

from parareal.parallel_scheme_training import parareal_scheme

os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

sys.path.append("..")
sys.path.append("../..")
import logging
import random

import numpy as np
import torch
from utils import (
    fetch_data_end_to_end,
    hyperparameter_grid_search_end_to_end,
    sample_label_normal_dist,
    setup_logger, flip_tensors,
)

from parareal.parareal_scheme_new import train_model_parareal
from models.model_end_to_end import save_model, setup_model
from models.param_settings import get_params, get_paths
from models.visualize_training import visualize_wavefield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_Dt_end_to_end(
    downsampling_model,
    upsampling_model,
    model_res,
    flipping,
    multi_step,
    logging_bool,
    visualize_res_bool,
    vis_save,
    experiment_index,
    weighted_loss,
):
    """
    Parameters
    ----------
    downsampling_model : defines which down sampling component to use
    upsampling_model : defines which up sampling component to use
    model_res : (int) resolution model can handle
    flipping : (bool) vertical and horizontal flipping for data augmentation
    multi_step : (bool) decides if multistep loss is used (see paper)
    logging_bool : (bool) decides if results are logged
    visualize_res_bool: (bool) decides if results are visualized
    vis_save : (bool) decides if visualization is saved
    experiment_index : (int) number of the experiment, explained in paper

    Returns
    -------
    trained end-to-end model
    """

    # params and logger setup
    (
        data_paths,
        train_logger_path,
        valid_logger_path,
        dir_path_save,
        vis_path,
        val_paths,
    ) = get_paths(experiment_index)
    param_d = get_params(experiment_index, flipping)
    param_d["model_res"], param_d["flipping"], param_d["multi_step"] = (
        model_res,
        flipping,
        multi_step,
    )
    model_name = f"{downsampling_model}_{upsampling_model}_{param_d['optimizer_name']}_{param_d['loss_function_name']}_{param_d['res_scaler']}_{model_res}_{flipping}_{multi_step}"
    train_logger, valid_logger, global_step = setup_logger(
        logging_bool,
        train_logger_path,
        valid_logger_path,
        model_name,
        model_res,
        vis_path,
    )
    logging.info(" ".join(["data settings:\n", ",\n".join(data_paths)]))
    logging.info(
        " ".join(
            ["param settings:\n", ",\n".join([f"{i}: {v}" for i, v in param_d.items()])]
        )
    )
    logging.info(model_name)
    logging.info(
        f"gpu available: {torch.cuda.is_available()} | number of gpus: {torch.cuda.device_count()}"
    )

    dict_best_score = {
        "score": float("inf"),
        "lr": None,
        "batch_size": None,
        "weight_decay": None,
        "dx": None,
        "dt": None,
    }

    logging.info(f"{'-'*100}\n\nHyperparameter Search:")

    # training and hyperparameter search with validation
    for lr, batch_size, weight_decay, dx, dt in hyperparameter_grid_search_end_to_end(
        upsampling_model, experiment_index, param_d
    ):
        if experiment_index == 6 or experiment_index == 4 or experiment_index == 5 or experiment_index == 7:
            batch_size //= 4

        param_d["c_delta_x"] = dx
        param_d["c_delta_t"] = dt

        logging.info(
            f"{'-'*10} Start training of {lr}, {batch_size}, {weight_decay}, {param_d['c_delta_x']:.5f}, {param_d['c_delta_t']:.5f} {'-'*10}"
        )

        model, optimizer, loss_f, label_distr_shift = setup_model(
            param_d,
            downsampling_model,
            upsampling_model,
            model_res,
            lr,
            weight_decay,
            device,
            weighted_loss,
            experiment_index,
        )

        train_loader, val_loader, _ = fetch_data_end_to_end(
            data_paths, batch_size, val_paths, experiment_index
        )

        for epoch in range(param_d["n_epochs"]):
            (
                train_loss_list,
                model,
                label_distr_shift,
                global_step,
                optimizer,
            ) = train_model(
                model,
                epoch,
                label_distr_shift,
                train_loader,
                param_d,
                global_step,
                logging_bool,
                optimizer,
                train_logger,
                loss_f,
                multi_step,
                weighted_loss,
                experiment_index,
                flipping
            )
            val_performance = val_model(
                model,
                logging_bool,
                train_logger,
                f"{vis_path}{model_name}_{lr}_{batch_size}_{weight_decay}_{dx}_{dt}",
                vis_save,
                global_step,
                valid_logger,
                val_loader,
                loss_f,
                epoch,
                train_loss_list,
                visualize_res_bool,
            )

            # for last iteration, check if performance is best so far
            if epoch + 1 == param_d["n_epochs"]:
                logging.info(
                    f"This iteration has a loss of {val_performance:.4f} compared to the best loss so far {dict_best_score['score']}"
                )
                if dict_best_score["score"] > val_performance:
                    dict_best_score = {
                        "score": val_performance,
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "dx": dx,
                        "dt": dt,
                    }

        save_model(
            model,
            f"{model_name}_{lr}_{batch_size}_{weight_decay}_{param_d['c_delta_x']:.5f}_{param_d['c_delta_t']:.5f}",
            dir_path_save + "validated_models/",
        )

    # testing
    logging.info(f"{'/'*100}\n\nTesting of best parameters: {dict_best_score}")

    param_d["c_delta_x"] = dict_best_score["dx"]
    param_d["c_delta_t"] = dict_best_score["dt"]

    model, optimizer, loss_f, label_distr_shift = setup_model(
        param_d,
        downsampling_model,
        upsampling_model,
        model_res,
        dict_best_score["lr"],
        dict_best_score["weight_decay"],
        device,
        weighted_loss,
        experiment_index,
    )
    train_loader, val_loader, test_loader = fetch_data_end_to_end(
        data_paths, dict_best_score["batch_size"], val_paths, experiment_index
    )
    trainval_loader = [
        d for dl in [train_loader, val_loader] for d in dl
    ]  # merge train and validation set for testing

    for epoch in range(param_d["n_epochs"]):
        train_loss_list, model, label_distr_shift, global_step, optimizer = train_model(
            model,
            epoch,
            label_distr_shift,
            trainval_loader,
            param_d,
            global_step,
            False,
            optimizer,
            train_logger,
            loss_f,
            multi_step,
            weighted_loss,
            experiment_index,
            flipping
        )
        _ = val_model(
            model,
            logging_bool,
            train_logger,
            f"{vis_path}best_{model_name}_{dict_best_score['lr']}_{dict_best_score['batch_size']}_{dict_best_score['weight_decay']}_{dict_best_score['dx']}_{dict_best_score['dt']}",
            vis_save,
            global_step,
            valid_logger,
            test_loader,
            loss_f,
            epoch,
            train_loss_list,
            visualize_res_bool,
        )

    save_model(
        model,
        f"best_{model_name}_{dict_best_score['lr']}_{dict_best_score['batch_size']}_{dict_best_score['weight_decay']}",
        dir_path_save,
    )

    logging.info("\n\n\n" + "*" * 100 + "\n\n\n")



def train_model(
    model,
    epoch,
    label_distr_shift,
    train_loader,
    param_d,
    global_step,
    logging_bool,
    optimizer,
    train_logger,
    loss_f,
    multi_step,
    weighted_loss,
    experiment_index,
    flipping
):
    if experiment_index == 6:

        return train_model_parareal(
            model,
            epoch,
            label_distr_shift,
            train_loader,
            param_d,
            global_step,
            logging_bool,
            optimizer,
            train_logger,
            loss_f,
            multi_step,
            weighted_loss,
        )
    else:
        model.train()
        train_loss_list = []
        if (epoch + 1) % 3 == 0 and weighted_loss:
            label_distr_shift += 1

        for i, data in enumerate(train_loader):
            loss_list = []
            data = data[0].to(device)  # b x n_snaps x 4 x w x h

            for input_idx in random.choices(
                range(param_d["n_snaps"] - 1), k=param_d["n_snaps"]
            ):
                label_range = sample_label_normal_dist(
                    input_idx,
                    param_d["n_snaps"],
                    label_distr_shift,
                    multi_step,
                    weighted_loss,
                )
                input_tensor = data[:, input_idx].detach()  # b x 4 x w x h

                v_flipped = False
                h_flipped = False

                print("before", input_idx)

                for label_idx in range(
                    input_idx + 1, label_range + 1
                ):  # randomly decide how long path is

                    print("---in", label_idx)

                    label = data[:, label_idx, :3]  # b x 3 x w x h

                    if flipping:
                        input_tensor, label, v_flipped, h_flipped = flip_tensors(input_tensor, label, v_flipped, h_flipped)

                    output = model(input_tensor)  # b x 3 x w x h
                    loss_list.append(loss_f(output, label))
                    input_tensor = torch.cat((output, input_tensor[:, 3].unsqueeze(dim=1)), dim=1)

            optimizer.zero_grad()
            sum(loss_list).backward()
            optimizer.step()

            if logging_bool:
                train_logger.add_scalar(
                    "loss", np.array(loss_list).mean(), global_step=global_step
                )
            train_loss_list.append(
                np.array([l.cpu().detach().numpy() for l in loss_list]).mean()
            )
            global_step += 1

        return train_loss_list, model, label_distr_shift, global_step, optimizer


def val_model(
    model,
    logging_bool,
    train_logger,
    vis_path,
    vis_save,
    global_step,
    valid_logger,
    val_loader,
    loss_f,
    epoch,
    train_loss_list,
    visualize_res_bool,
):

    if visualize_res_bool and not os.path.exists(vis_path):
        os.makedirs(vis_path)

    model.eval()
    with torch.no_grad():
        val_loss_list = []
        for i, data in enumerate(val_loader):
            n_snaps = data[0].shape[1]
            data = data[0].to(device)  # b x n_snaps x 3 x w x h

            visualize_list = []
            input_tensor = data[:, 0].clone()  # b x 4 x w x h
            vel = input_tensor[:, 3].unsqueeze(dim=1)

            for label_idx in range(1, n_snaps):
                label = data[:, label_idx, :3]  # b x 3 x w x h
                output = model(input_tensor)
                val_loss = loss_f(output, label)
                val_loss_list.append(val_loss.item())

                if i == 0 and visualize_res_bool:
                    # save only first element of batch
                    visualize_list.append(
                        (
                            val_loss.item(),
                            output[0].detach().cpu(),
                            label[0].detach().cpu(),
                        )
                    )
                input_tensor = torch.cat((output, vel), dim=1)

            if i == 0 and visualize_res_bool:
                visualize_wavefield(
                    epoch,
                    visualize_list,
                    input_tensor[0, 3].cpu(),
                    vis_save=vis_save,
                    vis_path=vis_path + "/",
                    initial_u=data[0, 0].unsqueeze(dim=0).cpu(),
                )

        val_performance = np.array(val_loss_list).mean()

        if logging_bool:
            train_logger.add_scalar(
                "loss", np.array(train_loss_list).mean(), global_step=global_step
            )
            valid_logger.add_scalar("loss", val_performance, global_step=global_step)

        if epoch % 1 == 0:
            logging.info(
                " ".join(
                    [
                        "epoch %d, train loss: %.5f, val loss: %.5f"
                        % (
                            epoch + 1,
                            np.array(train_loss_list).mean(),
                            np.array(val_loss_list).mean(),
                        )
                    ]
                )
            )

    return val_performance
