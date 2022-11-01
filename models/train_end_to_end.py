import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
import datetime
from model_end_to_end import restriction_nn
from model_utils import save_model, fetch_data_end_to_end
import torch
import torchvision.transforms.functional as TF
import random
import scipy.stats as ss

def train_Dt_end_to_end(batch_size = 1, lr = .001, res_scaler = 2, n_epochs = 500,
                        model_name = "unet", model_res = "128"):

    # model setup
    model = restriction_nn(res_scaler = res_scaler).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss() #nn.MSELoss()

    # data setup
    data_paths = [
        '../data/end_to_end_bp_m_200_' + str(model_res) + '_test.npz'
    ]
    train_loader, val_loader = fetch_data_end_to_end(data_paths, batch_size=batch_size)
    label_distr_shift = 0

    #TODO: tensorboard, proper analysis after, see neural networks course
    #TODO: read about batching how to do it best
    #TODO: figure out why nan for validation set if not flipping

    # training
    for epoch in range(n_epochs):

        # training
        model.train()
        train_loss_list = []
        for i, data in enumerate(train_loader):

            n_snaps = data[0].shape[1]
            data = data[0].to(device) # b x n_snaps x 3 x w x h

            if epoch % (n_epochs // n_snaps) == 0 and epoch != 0:
                label_distr_shift += 1

            for input_idx in random.choices(range(n_snaps-1), k=n_snaps-1):  # randomly shuffle order

                input_tensor = data[:, input_idx, :, :, :] # b x 3 x w x h
                h_flipped, v_flipped = False, False

                # randomly sample label idx from normal distribution
                possible_label_range = np.arange(input_idx + 2 - label_distr_shift, 12 - label_distr_shift)  # [a,b-1]
                prob = ss.norm.cdf(possible_label_range + 0.5, scale=3) - ss.norm.cdf(possible_label_range - 0.5, scale=3)
                label_range = list(np.random.choice(possible_label_range + label_distr_shift, size=1, p=prob / prob.sum()))[0]

                for label_idx in range(input_idx+1, label_range): # randomly decide how long path is

                    label = data[:, label_idx, :2, :, :] # b x 2 x w x h
                    if v_flipped: label = TF.vflip(label)
                    if h_flipped: label = TF.hflip(label)

                    #random horizontal and vertical flipping
                    if random.random() > 0.5:
                        h_flipped = not h_flipped
                        input_tensor = TF.hflip(input_tensor)
                        label = TF.hflip(label)
                    if random.random() > 0.5:
                        input_tensor = TF.vflip(input_tensor)
                        label = TF.vflip(label)

                    output = model(input_tensor)
                    optimizer.zero_grad()
                    loss = loss_f(output, label)

                    loss.backward()
                    optimizer.step()
                    input_tensor[:, :2, :, :] = output.detach()
                    train_loss_list.append(loss.item())


        # validation
        model.eval()
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


                    # # visualization
                    # fig1 = plt.figure(figsize=(15, 15))
                    # ax1 = fig1.add_subplot(3, 2, 1)
                    # ax1.imshow(input_tensor[:, 0, :, :].squeeze())
                    # ax2 = fig1.add_subplot(3, 2, 2)
                    # ax2.imshow(input_tensor[:, 1, :, :].squeeze())
                    # ax3 = fig1.add_subplot(3, 2, 3)
                    # ax3.imshow(label[:, 0, :, :].squeeze())
                    # ax4 = fig1.add_subplot(3, 2, 4)
                    # ax4.imshow(label[:, 1, :, :].squeeze())
                    # ax5 = fig1.add_subplot(3, 2, 5)
                    # ax5.imshow(output[:, 0, :, :].detach().squeeze())
                    # ax6 = fig1.add_subplot(3, 2, 6)
                    # ax6.imshow(output[:, 1, :, :].detach().squeeze())
                    # ax5.set_title(str(loss.item()) + "_" + str(input_idx) + "_" + str(label_idx), fontsize=20)
                    # plt.show()


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
    train_Dt_end_to_end(model_name = "end_to_end_"+model_name, model_res = model_res, res_scaler=int(res_scaler))
    end_time = time.time()

    print('training done:', (end_time - start_time))




