import sys
sys.path.append("..")
from models.param_settings import get_params, get_paths
import numpy as np
from utils import fetch_data_end_to_end, setup_logger, choose_optimizer, choose_loss_function, sample_label_normal_dist
import torch
import logging
from models.model_end_to_end import Model_end_to_end, save_model
import random
from models.visualize_training import visualize_wavefield


def train_Dt_end_to_end(
        downsampling_model,
        upsampling_model,
        optimizer_name,
        loss_function_name,
        res_scaler,
        model_res,
        flipping,
        multi_step,
        logging_bool=False,
        vis_save=True
):
    '''
    Parameters
    ----------
    downsampling_model : defines which down sampling component to use
    upsampling_model : defines which up sampling component to use
    optimizer_name : name of optimizer to adjust weights
    loss_function_name : name of loss function
    res_scaler : (int) scale factor by which input is up sampled (usually 2 or 4)
    model_res : (int) resolution model can handle
    flipping : (bool) vertical and horizontal flipping for data augmentation
    multi_step : (bool) decides if multistep loss is used (see paper)
    logging_bool : (bool) decides if results are logged
    vis_save : (bool) decides if visualization is saved

    Returns
    -------
    trained end-to-end model
    '''

    # params and logger setup
    data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths = get_paths(model_res)
    model_name = "_".join([downsampling_model, upsampling_model, optimizer_name, loss_function_name, str(res_scaler), str(model_res), str(flipping), str(multi_step)])
    param_dict = get_params()
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
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size, val_paths)

    # training
    label_distr_shift = 1
    logging.info(" ".join(["-" * 20, "start training", "-" * 20]))
    for epoch in range(n_epochs):

        model.train()
        train_loss_list = []
        if (epoch + 1) % 3 == 0: label_distr_shift += 1

        for i, data in enumerate(train_loader):
            loss_list = []
            n_snaps = data[0].shape[1]
            data = data[0].to(device)  # b x n_snaps x 4 x w x h
            for input_idx in random.choices(range(n_snaps - 2), k=n_snaps):
                label_range = sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, multi_step)
                input_tensor = data[:, input_idx].detach()  # b x 4 x w x h

                for label_idx in range(input_idx + 1, label_range):  # randomly decide how long path is
                    label = data[:, label_idx, :3]  # b x 3 x w x h
                    output = model(input_tensor)  # b x 3 x w x h
                    loss_list.append(loss_f(output, label))
                    input_tensor = torch.cat((output, input_tensor[:, 3].unsqueeze(dim=1)), dim=1)

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

                visualize_list = []
                input_tensor = data[:, 0].clone()  # b x 4 x w x h
                vel = input_tensor[:, 3].unsqueeze(dim=1)

                for label_idx in range(1, n_snaps):
                    label = data[:, label_idx, :3]  # b x 3 x w x h
                    output = model(input_tensor)
                    val_loss = loss_f(output, label)
                    val_loss_list.append(val_loss.item())

                    if i == 0:
                        # save only first element of batch
                        visualize_list.append((val_loss.item(), output[0].detach().cpu(),
                                               label[0].detach().cpu()))
                    input_tensor = torch.cat((output, vel), dim=1)

                if i == 0:
                    visualize_wavefield(epoch, visualize_list, input_tensor[0, 3].cpu(), vis_save=vis_save,
                                        vis_path=vis_path, initial_u=data[0, 0].unsqueeze(dim=0).cpu())

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


if __name__ == "__main__":

    train_Dt_end_to_end(
        downsampling_model="Interpolation",
        upsampling_model="UNet3",
        optimizer_name="AdamW",
        loss_function_name="SmoothL1Loss",
        res_scaler=2,
        model_res=128,
        flipping=False,
        multi_step=-1  # shifting normal distribution
    )
