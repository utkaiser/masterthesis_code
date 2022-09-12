import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import unet as unet
import torch.optim as optim
from model_utils import npdat2Tensor, save_model, load_model

def train(epochs = 1, batchsize = 128, lr = 1e-4, nlayer = 3, wf = 1, fine_coarse_scale = 4, continue_training = False):

    #model configuration
    model = unet.UNet(wf=wf, depth=nlayer, scale_factor=fine_coarse_scale).double()
    if continue_training: load_model('./NLModule_w1_3layer_data[22].pt',
                                     model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #training parameters
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()

    #set up training data
    data_paths = ['../data/training_data_13.npz']
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

        train_loaders.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputdata, outputdata),
                                                         batch_size=batchsize, shuffle=True, num_workers=1))

    for epoch in range(epochs):
        print("-"*20, str(epoch), "-"*20)

        training_loss = 0.0
        id_loss = 0.0

        for train_loader in train_loaders:
            for i, data in enumerate(tqdm(train_loader)):

                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                id_loss += loss_f(
                    nn.functional.upsample(inputs[:, :3, :, :], scale_factor=fine_coarse_scale, mode='bilinear'),
                    labels).item()

            if epoch % 1 == 0:
                print('[%d] training data loss: %.5f | coarse loss: %.5f ' %
                      (epoch + 1, training_loss, id_loss))

        save_model(model)

if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print('Training done ', (end_time - start_time))

