import numpy as np
from model_utils import save_model, fetch_data_end_to_end, flip_tensors, sample_label_random, \
    get_paths, get_params, setup_logger, choose_optimizer, choose_loss_function
import torch
import random
import logging
from models.model_end_to_end import Model_end_to_end
from analysis.visualize_results.visualize_training import visualize_wavefield


def train_Dt_end_to_end(downsampling_model, upsampling_model, optimizer_name, loss_function_name, res_scaler, model_res, flipping,
                        logging_bool=False, visualize=True, params="0", vis_save=True):

    # params and logger setup
    data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths = get_paths(model_res)
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

    optimizer = choose_optimizer(optimizer_name, model, lr)
    loss_f = choose_loss_function(loss_function_name)

    # data setup
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size, shuffle=True,
                                                     val_paths=val_paths)
    label_distr_shift = 0
    # training
    logging.info(" ".join(["-" * 20, "start training", "-" * 20]))
    for epoch in range(n_epochs):

        model.train()
        train_loss_list = []

        for i, data in enumerate(train_loader):

            n_snaps = data[0].shape[1]
            data = data[0].to(device)  # b x n_snaps x 4 x w x h

            loss_list = []

            if epoch % max(1,(n_epochs // n_snaps)) == 0 and epoch != 0: label_distr_shift += 1

            for input_idx in random.sample(range(n_snaps - 1), k=n_snaps - 1):

                input_tensor = data[:, input_idx, :, :, :]  # b x 4 x w x h
                h_flipped, v_flipped = False, False

                # randomly sample label idx from normal distribution
                label_range = sample_label_random(input_idx, label_distr_shift, epoch, n_snaps)

                for label_idx in range(input_idx + 1, label_range):  # randomly decide how long path is

                    label = data[:, label_idx, :3, :, :].to(device)  # b x 3 x w x h
                    if flipping: input_tensor, label, v_flipped, h_flipped = flip_tensors(input_tensor, label, v_flipped, h_flipped)
                    output = model(input_tensor)  # b x 3 x w x h
                    loss = loss_f(output, label)
                    loss_list.append(loss)
                    input_tensor = torch.cat((output, torch.unsqueeze(input_tensor[:, 3, :, :], dim=1)), dim=1)

            optimizer.zero_grad()
            sum(loss_list).backward()
            optimizer.step()

            if logging_bool: train_logger.add_scalar('loss', np.array(loss_list).mean(), global_step=global_step)
            train_loss_list.append(np.array([l.cpu().detach().numpy() for l in loss_list]).mean())
            global_step += 1

        # validation
        model.eval()
        with torch.no_grad():
            val_loss_list, visualize_list = [], []
            for i, data in enumerate(val_loader):

                n_snaps = data[0].shape[1]
                data = data[0].to(device)  # b x n_snaps x 3 x w x h
                input_tensor = data[:, 0, :, :, :]  # b x 4 x w x h

                for label_idx in range(1, n_snaps):
                    label = data[:, label_idx, :3, :, :]  # b x 3 x w x h

                    output = model(input_tensor)
                    val_loss = loss_f(output, label)
                    val_loss_list.append(val_loss.item())

                    if visualize and i == 0:
                        # save only first element of batch
                        visualize_list.append((val_loss.item(), output[0, :, :, :].detach(),
                                               label[0, :, :, :].detach()))

                    input_tensor = torch.cat((output, input_tensor[:, 3, :, :].unsqueeze(dim=1)), dim=1)

                if visualize and i == 0:
                    visualize_wavefield(epoch, visualize_list, input_tensor[0, 3, :, :], vis_save=vis_save, vis_path=vis_path)

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
        128,
        # 256,
    ]
    flipping = [
        False,
        # True
    ]

    counter = 0
    for d in downsampling_model:
        for u in upsampling_model:
            for o in optimizer:
                for l in loss_function:
                    for scale in res_scaler:
                        for res in model_res:
                            if (scale == 4 and res == 256) or (scale == 2 and res == 128):
                                for f in flipping:
                                    if counter >= -1:
                                        train_Dt_end_to_end(
                                            downsampling_model = d,
                                            upsampling_model = u,
                                            optimizer_name = o,
                                            loss_function_name = l,
                                            res_scaler = scale,
                                            model_res = res,
                                            flipping = f
                                        )
                                    counter += 1


if __name__ == "__main__":
    grid_search_end_to_end()
