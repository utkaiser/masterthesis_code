import time
import numpy as np
import torch
import torch.nn as nn
import unet
import tiramisu
import u_transformer
import torch.optim as optim
from model_utils import save_model, load_model, npdat2Tensor
import datetime
import sys
# import matplotlib.pyplot as plt
# from generate_data.WaveUtil import WaveSol_from_EnergyComponent
# from skimage.transform import resize

def train(epochs = 850, lr = .01, nlayer = 3, wf = 1,
          fine_coarse_scale = 2, continue_training = False, model_name = "unet"):

    batchsize = 32

    # model configuration
    if model_name == "unet":
        model = unet.UNet(wf=wf, depth=nlayer, scale_factor=fine_coarse_scale).double()
        if continue_training: load_model(model_name+"_1", model)
    elif model_name == "tiramisu":
        model = tiramisu.FCDenseNet().double()
        if continue_training: load_model(model_name+"_1", model)
    elif model_name == "u_trans":
        model = u_transformer.U_Transformer(in_channels=4, classes=3).double()
        if continue_training: load_model(model_name + "_1", model)
    else:
        raise NotImplementedError("model with name " + model_name + " not implemented")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    #model = nn.DataParallel(model).to(device) #parallel computing model
    model.to(device)

    # training setup
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss() #before: MSE loss
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.991)

    # training data setup
    data_paths = [
        # '../data/bp_m_10000_128.npz',
        # '../data/train_data_waveguide_128.npz',
        # '../data/train_data_inclusion_128.npz',
        #'../data/train_data_fig9_128.npz'
    ]
    #train_loaders = fetch_data(data_paths, batchsize)

    #training loop

    for epoch in range(epochs):

        loss_list = []
        id_loss_list = []

        for i in [14]:
            npz_PropS = np.load('../data/traindata_name' + str(i) + '.npz')

            inputdata = torch.stack((npdat2Tensor(npz_PropS['Ucx']),
                                     npdat2Tensor(npz_PropS['Ucy']),
                                     npdat2Tensor(npz_PropS['Utc']),
                                     npdat2Tensor(npz_PropS['vel'])),
                                    dim=1)

            outputdata = torch.stack((npdat2Tensor(npz_PropS['Ufx']),
                                      npdat2Tensor(npz_PropS['Ufy']),
                                      npdat2Tensor(npz_PropS['Utf'])),
                                     dim=1)

            wavedataset = torch.utils.data.TensorDataset(inputdata, outputdata)
            trainLoader = torch.utils.data.DataLoader(wavedataset, batch_size=batchsize, shuffle=True, num_workers=1)

        #for train_loader in train_loaders:
            for i, data in enumerate(trainLoader):
                inputs, labels = data[0].to(device), data[1].to(device) #parallel computing data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()

                # if loss.item() > 1:
                #     f, ax = plt.subplots(1, 3)
                #     f.set_figheight(5)
                #     f.set_figwidth(10)
                #     d = inputs[0][3, :, :].detach().numpy()
                #
                #     ax[0].imshow(d)
                #     ax[0].set_title(loss.item())
                #
                #     dx = 2.0 / 128.0
                #     a, b, c = inputs[0][0, :, :].detach().numpy(), inputs[0][1, :, :].detach().numpy(), inputs[0][2, :,:].detach().numpy()
                #     res1 = WaveSol_from_EnergyComponent(a, b, c, d, dx, 0)  # capital lambda dagger
                #     ax[1].imshow(res1[0]*dx*dx)
                #
                #     a, b, c = labels[0][0, :, :].detach().numpy(), labels[0][1, :, :].detach().numpy(), labels[0][2, :,:].detach().numpy()
                #     res2 = WaveSol_from_EnergyComponent(a, b, c, resize(d, (128, 128)), dx, 0)  # capital lambda dagger
                #     ax[2].imshow(res2[0]*dx*dx)
                #     f.show()


                loss_list.append(loss.item())

                id_loss_list.append(
                    loss_f(nn.functional.upsample(inputs[:, :3, :, :], scale_factor=fine_coarse_scale, mode='bilinear'),
                           labels).item()
                )

        scheduler.step()

        mean_loss = np.array(loss_list).mean()
        mean_id_loss = np.array(id_loss_list).mean()
        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f | coarse loss: %.5f , lr %.5f' %
                  (epoch + 1, mean_loss, mean_id_loss, optimizer.param_groups[0]["lr"]))

        with open('../results/run_1/loss_list_'+model_name+'.txt', 'a') as fd:
            fd.write(f'\n{loss_list}')

        if epoch % 50 == 0: #saves first models as a test
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
    model_name = "u_trans" #sys.argv[1]
    print("start training", model_name)
    train(model_name = model_name)
    end_time = time.time()
    print('training done:', (end_time - start_time))