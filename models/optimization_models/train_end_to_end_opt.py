import sys
from models.optimization_models.optimize_solution import model_optimization_solution
sys.path.append("..")
import numpy as np
from models.model_utils import save_model, fetch_data_end_to_end, \
    get_paths, get_params, setup_logger, choose_optimizer, choose_loss_function
import torch
import logging
from models.model_end_to_end import Model_end_to_end


def train_Dtp_end_to_end(downsampling_model, upsampling_model, optimizer_name, loss_function_name, res_scaler, model_res,
                        flipping, optimization_type, multi_step,
                        logging_bool=False, params="0", vis_save=True):

    # params and logger setup
    data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths = get_paths(model_res, optimization_type)
    model_name = "_".join([downsampling_model, upsampling_model, optimizer_name, loss_function_name, str(res_scaler), str(model_res), str(flipping)])
    param_dict = get_params(params)
    batch_size, lr, n_epochs, boundary_c, delta_t_star, f_delta_x, n_epochs_save_model = \
        param_dict["batch_size"], param_dict["lr"], param_dict["n_epochs"], param_dict["boundary_c"], \
        param_dict["delta_t_star"], param_dict["f_delta_x"], param_dict["n_epochs_save_model"]
    train_logger, valid_logger, global_step = setup_logger(logging_bool, train_logger_path, valid_logger_path,
                                                           model_name, model_res, vis_path)
    logging.info(" ".join(["data settings:", ", ".join(data_paths)]))
    logging.info(" ".join(["param settings:", ", ".join([i + ": " + str(v) for i, v in param_dict.items()])]))
    logging.info(model_name)

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(" ".join(["gpu available:", str(torch.cuda.is_available()), "| n of gpus:", str(torch.cuda.device_count())]))
    model = Model_end_to_end(param_dict, downsampling_model, upsampling_model, res_scaler, model_res).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model.load_state_dict(torch.load('../../results/run_3/good/saved_model_Interpolation_UNet3_AdamW_SmoothL1Loss_2_128_False_15.pt'))

    optimizer = choose_optimizer(optimizer_name, model, lr)
    loss_f = choose_loss_function(loss_function_name)

    # data setup
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size, shuffle=True,
                                                     val_paths=val_paths)

    # training
    label_distr_shift = 1
    logging.info(" ".join(["-" * 20, "start training", "-" * 20]))

    for epoch in range(n_epochs):
        model.train()
        train_loss_list = []
        if (epoch + 1) % 3 == 0: label_distr_shift += 1

        for i, data in enumerate(train_loader):
            n_snaps = data[0].shape[1]
            data = data[0].to(device)  # b x n_snaps x 4 x w x h
            loss_list = model_optimization_solution(data, model, loss_f, n_snaps, label_distr_shift, "train",
                                                    i, epoch, vis_path, vis_save, optimization_type, multi_step)

            optimizer.zero_grad()
            sum(loss_list).backward()
            optimizer.step()

            if logging_bool: train_logger.add_scalar('loss', np.array(loss_list).mean(), global_step=global_step)
            train_loss_list.append(np.array([l.cpu().detach().numpy() for l in loss_list]).mean())
            global_step += 1

        # validation
        model.eval()
        with torch.no_grad():
            val_loss_list = []
            for i, data in enumerate(val_loader):

                n_snaps = data[0].shape[1]
                data = data[0].to(device)  # b x n_snaps x 3 x w x h

                val_loss_list += model_optimization_solution(data, model, loss_f, n_snaps, -1, "val", i, epoch,
                                                             vis_path, vis_save, optimization_type, multi_step)

            if logging_bool:
                train_logger.add_scalar('loss', np.array(train_loss_list).mean(), global_step=global_step)
                valid_logger.add_scalar('loss', np.array(val_loss_list).mean(), global_step=global_step)

            if epoch % 1 == 0:
                logging.info(" ".join(
                    ['epoch %d, train loss: %.5f, test loss: %.5f' %
                     (epoch + 1, np.array(train_loss_list).mean(), np.array(val_loss_list).mean())]))

        if epoch % n_epochs_save_model == 0:  # saves first model as a test
            save_model(model, model_name + "_" + str(epoch), dir_path_save)
            model.to(device)

    save_model(model, model_name + "_final", dir_path_save)


def grid_search_end_to_end():

    downsampling_model = [
        "Interpolation",
        # "CNN",
        # "Simple"
    ]
    upsampling_model = [
        "UNet3",
        # "UNet6",
        # "Tiramisu",
        # "UTransform",
        # "Numerical_upsampling"
    ]
    optimizer = [
        "AdamW",
        # "RMSprop",
        # "SGD"
    ]
    loss_function = [
        "SmoothL1Loss",
        # "MSE"
    ]
    res_scaler = [
        2,
        # 4
    ]
    model_res = [
        # 128,
        256,
    ]
    flipping = [
        False,
        # True
    ]
    optimizations = [
        "parareal",
        # "procrustes",
    ]
    multi_step = [
        # 1,
        # 2,
        -1  # shifting normal distribution
    ]

    counter = 0
    for d in downsampling_model:
        for u in upsampling_model:
            for o in optimizer:
                for l in loss_function:
                    for scale in res_scaler:
                        for res in model_res:
                                for f in flipping:
                                    for opt in optimizations:
                                        for m in multi_step:
                                            train_Dtp_end_to_end(
                                                downsampling_model = d,
                                                upsampling_model = u,
                                                optimizer_name = o,
                                                loss_function_name = l,
                                                res_scaler = scale,
                                                model_res = res,
                                                flipping = f,
                                                optimization_type = opt,
                                                multi_step = m
                                            )
                                            counter += 1


if __name__ == "__main__":
    grid_search_end_to_end()
