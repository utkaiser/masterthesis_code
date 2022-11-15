import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import datetime
from model_end_to_end import Restriction_nn
from model_utils import save_model, fetch_data_end_to_end, flip_tensors, sample_label_random
import torch
import random
import torch.utils.tensorboard as tb
from visualize_progress import visualize_wavefield

def train_Dt_end_to_end(batch_size = 15, lr = .001, res_scaler = 2, n_epochs = 500,
                        model_name = "unet", model_res = "128", logging=False,
                        validate = False, flipping=False, visualize=False, boundary_c = 'periodic',
                        data_paths = ['../data/end_to_end_bp_m_200_2000.npz'],
                        train_logger_path = '../results/run_2/log_train/',
                        valid_logger_path = '../results/run_2/log_valid/'):

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

    model = Restriction_nn(res_scaler = res_scaler,boundary_c=boundary_c, delta_t_star=.06, f_delta_x = 2.0 / 128.0).double()
    model = torch.nn.DataParallel(model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr) #SGD(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss() #nn.MSELoss()

    # data setup
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size, shuffle=True, validate = validate)
    label_distr_shift = 0

    # training
    for epoch in range(n_epochs):
        print("epoch", epoch, "-"*20)

        # training
        model.train()
        train_loss_list = []
        for i, data in enumerate(train_loader):

            n_snaps = data[0].shape[1]
            data = data[0].to(device) # b x n_snaps x 4 x w x h
            loss_list = []

            if epoch % (n_epochs // n_snaps) == 0 and epoch != 0: label_distr_shift += 1

            for input_idx in random.choices(range(n_snaps-1), k=n_snaps-1):  # randomly shuffle order
            #for input_idx in range(1):
                if visualize: visualize_list = []

                input_tensor = data[:, input_idx, :, :, :] # b x 4 x w x h
                h_flipped, v_flipped = False, False

                # randomly sample label idx from normal distribution
                label_range = sample_label_random(input_idx, label_distr_shift)

                for label_idx in range(input_idx+1, label_range): # randomly decide how long path is

                    label = data[:, label_idx, :3, :, :].to(device) # b x 3 x w x h

                    if flipping:
                        input_tensor, label, v_flipped, h_flipped = flip_tensors(input_tensor, label, v_flipped, h_flipped)

                    output = model(input_tensor.to(device))  # b x 3 x w x h

                    loss = loss_f(output, label)
                    loss_list.append(loss)

                    if visualize:
                        # save only first element of batch
                        visualize_list.append((loss.item(), input_idx, label_idx, input_tensor[0,:,:,:].detach(), output[0,:,:,:].detach(), label[0,:,:,:].detach()))

                    input_tensor = torch.cat((output, torch.unsqueeze(input_tensor[:, 3, :, :], dim=1)), dim=1).detach()

                if visualize: visualize_wavefield(visualize_list,scaler=res_scaler)

            optimizer.zero_grad()

            sum(loss_list).backward()
            optimizer.step()
            print(sum(loss_list).item())

            if logging: train_logger.add_scalar('loss', sum(loss_list).item(), global_step=global_step)
            train_loss_list.append(sum(loss_list).item())
            global_step += 1

        if logging: train_logger.add_scalar('loss', np.array(train_loss_list).mean(), global_step=global_step)


        # validation
        if validate:
            model.eval()
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
            save_model(model, model_name + str(model_res))
            model.to(device)

    save_model(model, model_name + str(model_res))


if __name__ == "__main__":

    start_time = time.time()

    model_name = "unet" #sys.argv[1]
    model_res = "128" #sys.argv[2]
    res_scaler = "2" #sys.argv[3]
    print("start training", model_name, model_res)
    train_Dt_end_to_end(model_name = "end_to_end_"+model_name, model_res = model_res, res_scaler=int(res_scaler),
                        logging=False, visualize=True, boundary_c = 'absorbing', validate=False,
                        data_paths = ['../data/end_to_end_bp_m_200_2000.npz'])
    end_time = time.time()

    print('training done:', (end_time - start_time))


