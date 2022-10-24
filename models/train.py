import time
import numpy as np
import torch
import torch.nn as nn
import model_unet
import model_tiramisu
import model_u_transformer
import torch.optim as optim
from model_utils import save_model, load_model
import datetime
from model_utils import fetch_data
import torchvision.transforms.functional as TF
import random
#from torchsummary import summary
import sys

def train(epochs = 500, lr = .001, nlayer = 3, wf = 1, continue_training = False,
          model_name = "unet", batch_size = 64, gamma = 0.991, model_res ="128", scaler = 2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configuration
    if model_name == "unet":
        model = model_unet.UNet(depth=nlayer, wf=wf, acti_func='relu', scale_factor=scaler).double()
        model = nn.DataParallel(model).to(device)  # parallel computing model
    elif model_name == "tiramisu":
        model = model_tiramisu.FCDenseNet(scale_factor= scaler).double()
        batch_size = 20
        model = nn.DataParallel(model).to(device)  # parallel computing model
    elif model_name == "u_trans":
        batch_size = 20
        model = model_u_transformer.U_Transformer(in_channels=4, classes=3).double()
        model = nn.DataParallel(model).to(device)  # parallel computing model
    else:
        raise NotImplementedError("model with name " + model_name + " not implemented")

    if continue_training: load_model(model_name + "_1", model)
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    # summary(model, (4, int(resolution)//int(fine_coarse_scale), int(resolution)//int(fine_coarse_scale)))

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # training data setup
    data_paths = [
        '../data/bp_m_200_'+str(int(model_res)//scaler)+"_"+str(model_res)+'.npz'
    ]
    train_loaders = fetch_data(data_paths, batch_size)

    #training loop
    for epoch in range(epochs):

        loss_list = []
        id_loss_list = []

        for train_loader in train_loaders:
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device) #parallel computing data

                # random horizontal and vertical flipping
                if random.random() > 0.5:
                    inputs = TF.hflip(inputs)
                    labels = TF.hflip(labels)
                if random.random() > 0.5:
                    inputs = TF.vflip(inputs)
                    labels = TF.vflip(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                id_loss_list.append(
                    loss_f(nn.functional.upsample(inputs[:, :3, :, :], scale_factor=scaler, mode='bilinear'),
                           labels).item()
                )

        #scheduler.step()

        mean_loss = np.array(loss_list).mean()
        mean_id_loss = np.array(id_loss_list).mean()
        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f | coarse loss: %.5f' %  #, lr %.5f'
                  (epoch + 1, mean_loss, mean_id_loss)) # optimizer.param_groups[0]["lr"]

        with open('../results/run_1/loss_list_'+ model_name + model_res +'.txt', 'a') as fd:
            fd.write(f'\n{loss_list}')

        if epoch % 50 == 0: #saves first models as a test
            save_model(model, model_name + "_" + str(int(model_res) // scaler) + "_" + str(model_res))
            model.to(device)

    save_model(model, model_name + str(model_res))


if __name__ == "__main__":

    start_time = time.time()

    # model_name = "unet"
    # model_res = "128"
    # scaler = "2"

    model_name = sys.argv[1]
    model_res = sys.argv[2]
    scaler = sys.argv[3]

    print("start training", model_name, model_res)
    train(model_name = model_name, model_res = model_res, scaler=int(scaler))
    end_time = time.time()
    print('training done:', (end_time - start_time))