import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import unet
import tiramisu
import torch.optim as optim
from model_utils import npdat2Tensor, save_model, load_model
import sys

def train(epochs = 850, batchsize = 256, lr = 1e-4, nlayer = 3, wf = 1,
          fine_coarse_scale = 4, continue_training = False, model_name = "unet"):

    # model configuration
    if model_name == "unet":
        model = unet.UNet(wf=wf, depth=nlayer, scale_factor=fine_coarse_scale).double()
        if continue_training: load_model(model_name+"_1", model)
    elif model_name == "tiramisu":
        model = tiramisu.FCDenseNet().double()
        if continue_training: load_model(model_name+"_1", model)
    else:
        raise NotImplementedError("model with name " + model_name + " not implemented")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("gpu available:", torch.cuda.is_available())

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()

    # training data setup
    data_paths = ['../data/training_data_12.npz']
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
                                                   batch_size=batchsize, shuffle=True, num_workers=1)
        train_loaders.append(data_loader)


    #training loop

    for epoch in range(epochs):
        print("-"*20, str(epoch), "-"*20)

        loss_list = []
        id_loss_list = []

        for train_loader in train_loaders:
            for i, data in enumerate(tqdm(train_loader)):
                if i == 3:
                    break
                inputs, labels = data[0].to(device), data[1].to(device)
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
                print('[%d] training data loss: %.5f | coarse loss: %.5f ' %
                      (epoch + 1, mean_loss, mean_id_loss))

        with open('../data/loss_list_'+model_name+'.txt', 'a') as fd:
            fd.write(f'\n{str(epoch)}')
            fd.write(f'\n{loss_list}')

        if epoch % 200 == 0:
            save_model(model, model_name)
            model.to(device)

    save_model(model, model_name)

if __name__ == "__main__":

    start_time = time.time()
    model_name = "unet"#sys.argv[1]
    print("start training", model_name)
    train(model_name = model_name, epochs=1)
    end_time = time.time()
    print('Training done:', (end_time - start_time))