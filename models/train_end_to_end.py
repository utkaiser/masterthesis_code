import numpy as np
import time
from unet_end_to_end import restriction_nn
import torch.optim as optim
import torch.nn as nn
import datetime
from model_utils import save_model, fetch_data_end_to_end
import torch

def train_end_to_end(batch_size = 1, lr = .01, gamma = .991, fine_coarse_scale = 2, n_epochs = 500,
                         nlayer = 3, wf = 1, continue_training = False, model_name = "unet", resolution = "128"):


    ### models ###
    model = restriction_nn(down_factor=fine_coarse_scale).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


    ########### approach 1: D_t ##################

    ### data ###
    data_paths = [
        '../data/end_to_end_bp_m_200_' + str(resolution) + '.npz',
    ]
    train_loaders = fetch_data_end_to_end(data_paths,batchsize = batch_size)


    ### training ###
    for epoch in range(n_epochs):
        loss_list = []
        for train_loader in train_loaders:
            for i, data in enumerate(train_loader):

                inputs, labels = data[0].to(device), data[1].to(device)  # parallel computing data

                outputs = model(inputs)

                optimizer.zero_grad()
                loss = loss_f(outputs, labels) #fine solution as target
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()

        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f' % (epoch + 1, np.array(loss_list).mean()))

        if epoch % 50 == 0:  # saves first models as a test
            save_model(model, model_name + str(resolution))
            model.to(device)

    save_model(model, model_name + str(resolution))


if __name__ == "__main__":

    start_time = time.time()
    model_name = "unet"#sys.argv[1]
    model_resolution = "128"#sys.argv[2]
    scaler = "2"#sys.argv[3]
    print("start training", model_name, model_resolution)
    train_end_to_end(model_name = model_name, resolution = model_resolution, fine_coarse_scale=int(scaler))
    end_time = time.time()
    print('training done:', (end_time - start_time))



'''
    #### approach 2: D_t^p (parareal scheme) ####

    ### data ###
    train_loaders = fetch_data('../data/bp_m_200_128.npz',
                      batchsize=batch_size,
                      shuffle=True)

    ### training ###
    for epoch in range(n_epochs):
        loss_list = []
        for train_loader in train_loaders:
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device) #parallel computing data
       


                optimizer.zero_grad()
                loss = loss_f(outputs, labels)  # fine solution as target
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
    
        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f' % (epoch + 1, np.array(loss_list).mean()))

        if epoch % 50 == 0:  # saves first models as a test
            save_model(model, model_name)
            model.to(device)

    save_model(model, model_name)
'''




