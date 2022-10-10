import time
import numpy as np
import torch
import torch.nn as nn
import unet_old as unet
import tiramisu
import u_transformer
import torch.optim as optim
from model_utils import save_model, load_model
import datetime
from model_utils import fetch_data
import sys

def train(epochs = 500, lr = .001, nlayer = 3, wf = 1,
          fine_coarse_scale = 2, continue_training = False,
          model_name = "unet", batchsize = 32, gamma = 0.991, resolution = "128"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configuration
    if model_name == "unet":
        model = unet.UNet(depth=6, wf=1, acti_func='relu', scale_factor=fine_coarse_scale).double()
        if continue_training: load_model(model_name+"_1", model)
    elif model_name == "tiramisu":
        model = tiramisu.FCDenseNet(scale_factor= fine_coarse_scale).double()
        batchsize = 20
        if continue_training: load_model(model_name+"_1", model)
        model = nn.DataParallel(model).to(device)  # parallel computing model
    elif model_name == "u_trans":
        model = u_transformer.U_Transformer(in_channels=4, classes=3).double()
        if continue_training: load_model(model_name + "_1", model)
    else:
        raise NotImplementedError("model with name " + model_name + " not implemented")

    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # training data setup
    data_paths = [
        '../data/bp_m_200_'+str(resolution)+'.npz',
    ]
    train_loaders = fetch_data(data_paths, batchsize)

    #training loop
    for epoch in range(epochs):

        loss_list = []
        id_loss_list = []

        for train_loader in train_loaders:
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device) #parallel computing data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

                id_loss_list.append(
                    loss_f(nn.functional.upsample(inputs[:, :3, :, :], scale_factor=fine_coarse_scale, mode='bilinear'),
                           labels).item()
                )

        #scheduler.step()

        mean_loss = np.array(loss_list).mean()
        mean_id_loss = np.array(id_loss_list).mean()
        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f | coarse loss: %.5f' %  #, lr %.5f'
                  (epoch + 1, mean_loss, mean_id_loss)) # optimizer.param_groups[0]["lr"]

        with open('../results/run_1/loss_list_'+ model_name + resolution +'.txt', 'a') as fd:
            fd.write(f'\n{loss_list}')

        if epoch % 50 == 0: #saves first models as a test
            save_model(model, model_name + str(resolution))
            model.to(device)

    save_model(model, model_name + str(resolution))


if __name__ == "__main__":

    start_time = time.time()
    model_name = sys.argv[1]
    model_resolution = sys.argv[2]
    scaler = sys.argv[3]
    print("start training", model_name, model_resolution)
    train(model_name = model_name, resolution = model_resolution, fine_coarse_scale=int(scaler))
    end_time = time.time()
    print('training done:', (end_time - start_time))