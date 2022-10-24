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

def train_Dt_end_to_end(batch_size = 1, lr = .001, res_scaler = 2, n_epochs = 500,
                        model_name = "unet", model_res = "128"):

    # model setup
    model = restriction_nn(res_scaler = res_scaler).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()

    # data setup
    data_paths = [
        '../data/end_to_end_bp_m_200_' + str(model_res) + '.npz',
    ]
    train_loaders = fetch_data_end_to_end(data_paths, batch_size = batch_size)

    # training
    for epoch in range(n_epochs):
        loss_list = []
        for train_loader in train_loaders:
            for i, data in enumerate(train_loader):

                n_snaps = data[0].shape[1]
                data = data[0].to(device) # b x n_snaps x 3 x w x h

                for input_idx in range(n_snaps-1):
                    input = data[:, input_idx, :, :, :] # b x 3 x w x h

                    for label_idx in range(input_idx+1, n_snaps):

                        label = data[:, label_idx, :2, :, :] # b x 2 x w x h

                        # random horizontal and vertical flipping
                        if random.random() > 0.5:
                            input = TF.hflip(input)
                            label = TF.hflip(label)
                        if random.random() > 0.5:
                            input = TF.vflip(input)
                            label = TF.vflip(label)

                        output = model(input)

                        optimizer.zero_grad()
                        loss = loss_f(output, label)
                        loss_list.append(loss.item())
                        loss.backward()
                        optimizer.step()

                        input[:, :2,:,:] = output.detach()


        if epoch % 1 == 0:
            print(datetime.datetime.now().strftime("%H:%M:%S"), 'epoch %d loss: %.5f' % (epoch + 1, np.array(loss_list).mean()))

        if epoch % 50 == 0:  # saves first models as a test
            save_model(model, model_name + str(model_res))
            model.to(device)

    save_model(model, model_name + str(model_res))


if __name__ == "__main__":

    start_time = time.time()

    model_name = "unet" #sys.argv[1]
    model_res = "256" #sys.argv[2]
    res_scaler = "4" #sys.argv[3]
    print("start training", model_name, model_res)
    train_Dt_end_to_end(model_name = "end_to_end_"+model_name, model_res = model_res, res_scaler=int(res_scaler))
    end_time = time.time()

    print('training done:', (end_time - start_time))




