import time
import numpy as np
import torch
import torch.nn as nn
import unet
import tiramisu as tiramisu
import torch.optim as optim
from model_utils import save_model, load_model, npdat2Tensor
import datetime

def train(epochs = 850, lr = 1e-4, nlayer = 3, wf = 1,
          fine_coarse_scale = 4, continue_training = False, model_name = "unet"):

    batchsize = 256 if model_name == "unet" else 256 #otherwise uses too much memory

    # model configuration
    if model_name == "unet":
        model = unet.UNet(wf=wf, depth=nlayer, scale_factor=fine_coarse_scale).double()
        if continue_training: load_model(model_name+"_1", model)
    elif model_name == "tiramisu":
        model = tiramisu.FCDenseNet().double()
        if continue_training: load_model(model_name+"_1", model)
    else:
        raise NotImplementedError("model with name " + model_name + " not implemented")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model = nn.DataParallel(model).to(device) #parallel computing model

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()

    # training data setup
    data_paths = [
        '../data/train_data_fig9_128.npz',
        '../data/train_data_waveguide_128.npz',
        '../data/train_data_inclusion_128.npz',
        '../data/train_data_fig9_128.npz'
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

                id_loss_list.append(loss_f(
                    nn.functional.upsample(inputs[:, :3, :, :], scale_factor=fine_coarse_scale, mode='bilinear'),
                    labels).item()
                )

            mean_loss = np.array(loss_list).mean()
            mean_id_loss = np.array(id_loss_list).mean()
            if epoch % 1 == 0:
                print(datetime.datetime.now(), 'epoch %d: loss: %.5f | coarse loss: %.5f ' %
                      (epoch + 1, mean_loss, mean_id_loss))

        with open('../data/loss_list_'+model_name+'.txt', 'a') as fd:
            fd.write(f'\n{loss_list}')

        if epoch % 100 == 0: #saves first models as a test
            save_model(model, model_name)
            model.to(device)

    save_model(model, model_name)

def fetch_data(data_paths, batchsize, shuffle=True):
    print("setting up data")

    total_n_datapoints = 0
    train_loaders = []
    for path in data_paths:
        npz_PropS = np.load(path)
        inputdata = torch.stack((npdat2Tensor(npz_PropS['Ucx']),
                                 npdat2Tensor(npz_PropS['Ucy']),
                                 npdat2Tensor(npz_PropS['Utc']),
                                 npdat2Tensor(npz_PropS['vel'])), dim=1)
        outputdata = torch.stack((npdat2Tensor(npz_PropS['Ufx']),
                                  npdat2Tensor(npz_PropS['Ufy']),
                                  npdat2Tensor(npz_PropS['Utf'])), dim=1)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputdata, outputdata),
                                                  batch_size=batchsize, shuffle=shuffle, num_workers=1)
        total_n_datapoints += len(data_loader)
        train_loaders.append(data_loader)

    print("total number of data points:", total_n_datapoints * batchsize)

    return train_loaders

if __name__ == "__main__":

    start_time = time.time()
    model_name = "tiramisu"#sys.argv[1]
    print("start training", model_name)
    train(model_name = model_name)
    end_time = time.time()
    print('training done:', (end_time - start_time))