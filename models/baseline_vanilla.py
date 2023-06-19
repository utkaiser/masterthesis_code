from models.model_end_to_end import Model_end_to_end
from models.param_settings import get_params
import numpy as np
from models.utils import choose_loss_function, vanilla_benchmark_grid_search_end_to_end, fetch_data_end_to_end
import logging
import sys
import os
import torch


def get_results_vanilla(
        experiment_index = 0,
        flipping = False,
        model_res = 128
):

    save_path = f"../results/run_5/exp{experiment_index}/vanilla_baseline/"
    setup_logger_vanilla(save_path)
    data_paths, val_paths = get_vanilla_paths(experiment_index)

    param_d = get_params(experiment_index, flipping)  # get all relevant parameters
    loss_f = choose_loss_function(param_d["loss_function_name"])  # get loss function
    train_loader, val_loader, test_loader = fetch_data_end_to_end(data_paths, 1, val_paths)  # get dataloaders

    # keeps track of score board during parameter analysis (experiment 2)
    dict_best_score = {
        "score": float("inf"),
        "lr": None,
        "batch_size": None,
        "weight_decay": None
    }

    # hyperparameter analysis
    for dx, dt in vanilla_benchmark_grid_search_end_to_end(param_d,experiment_index):

        param_d["c_delta_x"] = dx
        param_d["c_delta_t"] = dt

        model = Model_end_to_end(param_d, "Interpolation", "Interpolation", model_res).double()  # build vanilla baseline model
        val_performance = val_model_vanilla(model, val_loader, loss_f)

        logging.info(f"Baseline_{param_d['c_delta_x']:.5f}_{param_d['c_delta_t']:.5f} has {val_performance} compared to best performance so far {dict_best_score['score']}")

        if dict_best_score["score"] > val_performance:
            dict_best_score = {
                "score": val_performance,
                "dx": dx,
                "dt": dt
            }
    logging.info(f"Best overall performance is {dict_best_score['score']} using baseline_{dict_best_score['dx']:.5f}_{dict_best_score['dt']:.5f}")


    # testing
    param_d["c_delta_x"] = dict_best_score["dx"]
    param_d["c_delta_t"] = dict_best_score["dt"]
    model = Model_end_to_end(param_d, "Interpolation", "Interpolation", model_res).double() # build vanilla baseline model
    trainval_loader = [d for dl in [train_loader, val_loader] for d in dl]  # merge train and validation set for testing

    val_performance = val_model_vanilla(model, trainval_loader, loss_f)

    logging.info(f"Test performance is {val_performance} using baseline_{param_d['c_delta_x']:.5f}_{param_d['c_delta_t']:.5f}")



def val_model_vanilla(
        model,
        val_loader,
        loss_f
):

    val_loss_list = []

    for i, data in enumerate(val_loader):

        n_snaps = data[0].shape[1]
        data = data[0]

        input_tensor = data[:, 0].clone()  # b x 4 x w x h
        vel = input_tensor[:, 3].unsqueeze(dim=1)

        for label_idx in range(1, n_snaps):
            label = data[:, label_idx, :3]  # b x 3 x w x h
            output = model(input_tensor)[:,:3]
            val_loss = loss_f(output, label)
            val_loss_list.append(val_loss.item())
            input_tensor = torch.cat((output, vel), dim=1)

    val_performance = np.array(val_loss_list).mean()

    return val_performance



def setup_logger_vanilla(save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(save_path + '.log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



def get_vanilla_paths(
        experiment_index
):

    # todo: change back

    data_paths = [
        # 12.5 * 400 * 8 = 40,000
        '../data/Dt_128/bp_marmousi_128_400_none_0.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_1.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_2.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_3.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_4.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_5.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_6.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_7.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_8.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_9.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_10.npz',
        '../data/Dt_128/bp_marmousi_128_400_none_11.npz',
        '../data/Dt_128/bp_marmousi_128_200_none_12.npz',
    ]

    test_paths = [
        '../data/Dt_128/test_128_84_none_0.npz',
    ]

    return data_paths, test_paths


if __name__ == '__main__':
    get_results_vanilla(
        experiment_index = 0
    )


