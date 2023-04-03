import sys
sys.path.append("")
sys.path.append("..")
import numpy as np
from models.utils import setup_logger, fetch_data_end_to_end
from parareal.visualize_training import visualize_wavefield
from parareal.param_settings import get_paths, get_params
import torch
import logging
from models.model_end_to_end import Model_end_to_end, save_model
from parareal.parallel_scheme import optimize_solution


def train_Dt_end_to_end(logging_bool=False, visualize=True, vis_param=1, params="0", vis_save=True):
    # params and logger setup
    data_paths, train_logger_path, valid_logger_path, dir_path_save, vis_path, val_paths = get_paths()
    param_dict = get_params(params)
    batch_size, lr, res_scaler, n_epochs, model_name, model_res, flipping, boundary_c, delta_t_star, f_delta_x, n_epochs_save_model = \
        param_dict["batch_size"], param_dict["lr"], param_dict["res_scaler"], param_dict["n_epochs"], param_dict[
            "model_name"], \
        param_dict["model_res"], param_dict["flipping"], param_dict["boundary_c"], param_dict["delta_t_star"], \
        param_dict["f_delta_x"], param_dict["n_epochs_save_model"]
    train_logger, valid_logger, global_step = setup_logger(logging_bool, train_logger_path, valid_logger_path,
                                                           model_name, model_res, vis_path)
    logging.info(" ".join(["data settings:", ", ".join(data_paths)]))
    logging.info(" ".join(["param settings:", ", ".join([i + ": " + str(v) for i, v in param_dict.items()])]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(" ".join(["gpu available:", str(torch.cuda.is_available()), "| n of gpus:", str(torch.cuda.device_count())]))
    model = Model_end_to_end(param_dict, "Interpolation", "UNet3", res_scaler, model_res).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model.load_state_dict(torch.load('../results/run_3/good_one/saved_model_Interpolation_UNet3_AdamW_SmoothL1Loss_2_128_False_15.pt'))

    # data setup
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size, val_paths)
    label_distr_shift = 0
    loss_f = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    # training
    logging.info(" ".join(["-" * 20, "start training", "-" * 20]))
    for epoch in range(n_epochs):

        model.train()
        train_loss_list = []

        if (epoch + 1) % 3 == 0: label_distr_shift += 1

        for i, data in enumerate(train_loader):

            data = data[0].to(device)  # b x n_snaps x 4 x 256 x 256
            loss_list = optimize_solution(model, loss_f, data, label_distr_shift)  # b x 3 x w x h

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
                if i == 10: break
                n_snaps = data[0].shape[1]
                data = data[0].to(device)  # b x n_snaps x 3 x w x h

                for input_idx in range(n_snaps - 1):
                    input_tensor = data[:, input_idx, :, :, :]  # b x 4 x w x h

                    if visualize and (epoch + 1) % vis_param == 0 and i == 0 and input_idx == 0: visualize_list = []

                    for label_idx in range(input_idx + 1, n_snaps):
                        label = data[:, label_idx, :3, :, :]  # b x 3 x w x h

                        output = model(input_tensor)
                        val_loss = loss_f(output, label)
                        val_loss_list.append(val_loss.item())

                        if visualize and (epoch + 1) % vis_param == 0 and i == 0 and input_idx == 0:
                            # save only first element of batch
                            visualize_list.append((epoch, val_loss.item(), input_idx, label_idx,
                                                   input_tensor[0, :, :, :].detach(), output[0, :, :, :].detach(),
                                                   label[0, :, :, :].detach()))

                        input_tensor = torch.cat((output, torch.unsqueeze(input_tensor[:, 3, :, :], dim=1)), dim=1)

                    if visualize and (epoch + 1) % vis_param == 0 and i == 0 and input_idx == 0:
                        visualize_wavefield(visualize_list, scaler=res_scaler, vis_save=vis_save, vis_path=vis_path)

            if logging_bool:
                train_logger.add_scalar('loss', np.array(train_loss_list).mean(), global_step=global_step)
                valid_logger.add_scalar('loss', np.array(val_loss_list).mean(), global_step=global_step)

            if epoch % 1 == 0:
                logging.info(" ".join(
                    ['epoch %d, train loss: %.5f, test loss: %.5f' %
                     (epoch + 1, np.array(train_loss_list).mean(), np.array(val_loss_list).mean())]))

        if epoch % n_epochs_save_model == 0:  # saves first model as a test
            save_model(model, model_name + str(model_res), dir_path_save)
            model.to(device)

    save_model(model, model_name + str(model_res), dir_path_save)


if __name__ == "__main__":
    train_Dt_end_to_end()