import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import datetime
from model_end_to_end import Restriction_nn
from model_utils import save_model, fetch_data_end_to_end, flip_tensors, sample_label_random, visualize_wavefield, get_paths, get_params
import torch
import random
import torch.utils.tensorboard as tb

#TODO: get one step after init to work one elapse
#TODO: get two steps after init to work one elapse
#TODO: get any step after init to work one elapse
#TODO: get one step after init to work two elapse
#TODO: get any step after init to work two elapse
#TODO: get randomized approach to work
#goal: .00x

def train_Dt_end_to_end(logging=False, validate = False, visualize=False, vis_param=20, params="0"):

    data_paths, train_logger_path, valid_logger_path, dir_path_save = get_paths()
    batch_size, lr, res_scaler, n_epochs, model_name, model_res, flipping, boundary_c, delta_t_star, f_delta_x = get_params(params)

    #logger setup
    train_logger, valid_logger = None, None
    if logging:
        train_logger = tb.SummaryWriter(train_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
        valid_logger = tb.SummaryWriter(valid_logger_path + model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
    global_step = 0

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())

    model = Restriction_nn(res_scaler = res_scaler,boundary_c=boundary_c, delta_t_star=delta_t_star, f_delta_x = f_delta_x).double()
    model = torch.nn.DataParallel(model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr) #SGD(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss() #nn.MSELoss()

    # data setup
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size, shuffle=True, validate = validate)
    label_distr_shift = 0

    # training
    print("-"*20,"start training", "-"*20)
    for epoch in range(n_epochs):

        # training
        model.train()
        train_loss_list = []
        for i, data in enumerate(train_loader):

            n_snaps = data[0].shape[1]
            data = data[0].to(device) # b x n_snaps x 4 x w x h
            loss_list = []

            if epoch % (n_epochs // n_snaps) == 0 and epoch != 0: label_distr_shift += 1

            #for input_idx in random.choices(range(n_snaps-1), k=1):  # randomly shuffle order
            for input_idx in range(10):
                if visualize and epoch % vis_param == 0: visualize_list = []

                input_tensor = data[:, input_idx, :, :, :] # b x 4 x w x h
                h_flipped, v_flipped = False, False

                # randomly sample label idx from normal distribution
                label_range = sample_label_random(input_idx, label_distr_shift)

                for label_idx in range(input_idx+1, input_idx+2): # randomly decide how long path is

                    label = data[:, label_idx, :3, :, :].to(device) # b x 3 x w x h

                    if flipping:
                        input_tensor, label, v_flipped, h_flipped = flip_tensors(input_tensor, label, v_flipped, h_flipped)

                    output = model(input_tensor.to(device))  # b x 3 x w x h

                    loss = loss_f(output, label)
                    loss_list.append(loss)

                    if visualize and epoch % vis_param == 0:
                        # save only first element of batch
                        visualize_list.append((loss.item(), input_idx, label_idx, input_tensor[0,:,:,:].detach(), output[0,:,:,:].detach(), label[0,:,:,:].detach()))

                    input_tensor = torch.cat((output, torch.unsqueeze(input_tensor[:, 3, :, :], dim=1)), dim=1).detach()

                if visualize and epoch % vis_param == 0: visualize_wavefield(visualize_list,scaler=res_scaler, save=True)

            optimizer.zero_grad()
            sum(loss_list).backward()
            optimizer.step()

            if logging: train_logger.add_scalar('loss', sum(loss_list).item(), global_step=global_step)
            train_loss_list.append(np.array([a.detach().numpy() for a in loss_list]).mean())
            global_step += 1
        print("epoch %d | loss: %.5f" % (epoch, np.array(train_loss_list).mean()))
        if logging: train_logger.add_scalar('loss', np.array(train_loss_list).mean(), global_step=global_step)


        # validation
        if validate:
            with torch.no_grad():
                val_loss_list = []
                for i, data in enumerate(val_loader):
                    n_snaps = data[0].shape[1]
                    data = data[0].to(device)  # b x n_snaps x 3 x w x h

                    for input_idx in range(n_snaps - 1):
                        input_tensor = data[:, input_idx, :, :, :]  # b x 3 x w x h

                        for label_idx in range(input_idx + 1, n_snaps):
                            label = data[:, label_idx, :2, :, :]  # b x 2 x w x h
                            output = model(input_tensor)
                            val_loss = loss_f(output, label)

                            val_loss_list.append(val_loss.item())
                            input_tensor[:, :2, :, :] = output

            if logging: valid_logger.add_scalar('loss', np.array(val_loss_list).mean() ,global_step=global_step)

            if epoch % 1 == 0:
                print(datetime.datetime.now().strftime("%H:%M:%S"), 'epoch %d , train loss: %.5f, test loss: %.5f' %
                      (epoch + 1, np.array(train_loss_list).mean(), np.array(val_loss_list).mean()))

        if epoch % 50 == 0:  # saves first model as a test
            save_model(model, model_name + str(model_res), dir_path_save)
            model.to(device)

    save_model(model, model_name + str(model_res), dir_path_save)


if __name__ == "__main__":

    train_Dt_end_to_end()


