import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import datetime
from model_end_to_end import restriction_nn
from model_utils import save_model, fetch_data_end_to_end, flip_tensors
import torch
import random
import scipy.stats as ss
import torch.utils.tensorboard as tb
from visualize_progress import visualize_wavefield

def train_Dt_end_to_end(batch_size = 50, lr = .001, res_scaler = 2, n_epochs = 500,
                        model_name = "unet", model_res = "128", logging=False, validate = False, flipping=False, visualize=False):

    #logger setup
    train_logger, valid_logger = None, None
    if logging:
        train_logger = tb.SummaryWriter('../results/run_2/log_train/'+ model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
        valid_logger = tb.SummaryWriter('../results/run_2/log_valid/'+ model_name + str(model_res)
                                        + '/{}'.format(time.strftime('%H-%M-%S')) + '_test.npz', flush_secs=1)
    global_step = 0

    # model setup
    model = restriction_nn(res_scaler = res_scaler).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr) #SGD(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss() #nn.MSELoss()

    # data setup
    data_paths = [
        '../data/end_to_end_bp_m_200_' + str(model_res) + '.npz'
    ]
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size, shuffle=True)
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

            if epoch % (n_epochs // n_snaps) == 0 and epoch != 0:
                label_distr_shift += 1

            #for input_idx in random.choices(range(n_snaps-1), k=3):  # randomly shuffle order TODO: change back k to n_snaps-1
            for input_idx in range(1):
                if visualize: visualize_list = []

                input_tensor = data[:, input_idx, :, :, :] # b x 4 x w x h
                h_flipped, v_flipped = False, False

                # randomly sample label idx from normal distribution
                possible_label_range = np.arange(input_idx + 2 - label_distr_shift, 12 - label_distr_shift)  # [a,b-1]
                prob = ss.norm.cdf(possible_label_range + 0.5, scale=3) - ss.norm.cdf(possible_label_range - 0.5, scale=3)
                label_range = list(np.random.choice(possible_label_range + label_distr_shift, size=1, p=prob / prob.sum()))[0]

                for label_idx in range(input_idx+1, input_idx+2): # randomly decide how long path is

                    label = data[:, label_idx, :3, :, :] # b x 3 x w x h

                    if flipping:
                        input_tensor, label, v_flipped, h_flipped = flip_tensors(input_tensor, label, v_flipped, h_flipped)

                    output = model(input_tensor)

                    loss = loss_f(output, input_tensor)#label)
                    loss_list.append(loss)

                    if visualize:
                        # save only first element of batch
                        visualize_list.append((loss, input_idx, label_idx, input_tensor[0,:,:,:].detach(), output[0,:,:,:].detach(), label[0,:,:,:].detach()))

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

        if epoch % 20 == 0:  # saves first model as a test
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
                        logging=False, visualize=True, absorbing_bc = False)
    end_time = time.time()

    print('training done:', (end_time - start_time))


