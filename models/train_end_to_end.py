import numpy as np
import generate_data.wave2 as w2
from model_utils import fetch_data
from unet_end_to_end import restriction_nn
import torch.optim as optim
import torch.nn as nn
import datetime
from model_utils import save_model
import torch

def propagate_end_to_end(model_name = "unet"):

    ### parameter setup ###
    Tf = 2.0
    cT = 0.2
    dx = 2.0 / 128.0
    dt = dx / 20
    m = 2
    rt = 4
    mt = round(Tf / cT)
    ny, nx = 64, 64
    x = np.arange(-1, 1, dx)
    y = np.arange(-1, 1, dx)
    xx, yy = np.meshgrid(x, y)

    #training params
    batch_size = 32
    lr = .01
    gamma = .991
    fine_coarse_scale = 2
    n_epochs = 500
    nlayer = 3
    wf = 1
    continue_training = False

    ### models ###
    model = restriction_nn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("gpu available:", torch.cuda.is_available(), "| n of gpus:", torch.cuda.device_count())
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


    ########### approach 1: D_t ##################

    ### data ###
    np.random.seed = 21  # TODO: randomize
    center = np.array([-0.8, -0.8])
    vel = 1. + 0.0 * yy - 0.5 * (np.abs(yy + xx - 0.) > 0.4) + 0. * (np.abs(xx - 0.4) < 0.2) * (np.abs(yy - 0.5) < 0.1)
    u0 = u_prev = np.exp(-250.0 * (0.2 * (xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * np.cos(
        8 * np.pi * (yy - center[1]))
    ut_prev = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])

    ### training ###
    for epoch in range(n_epochs):
        loss_list = []
        for j in range(1,mt):

            labels = w2.velocity_verlet_time_integrator(
                u_prev,
                ut_prev,
                vel,
                dx,dt,cT
            )

            outputs = model(u_prev,
                            ut_prev,
                            vel,
                            cT, dx*m, dt*rt, u0, dx)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels) #fine solution as target
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            u_prev, ut_prev = labels # fine solution as input for next iteration

        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f' % (epoch + 1, np.array(loss_list).mean()))

        if epoch % 50 == 0:  # saves first models as a test
            save_model(model, model_name)
            model.to(device)

    save_model(model, model_name)




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
        for j in range(1, mt):
            labels = w2.velocity_verlet_time_integrator(
                u_prev,
                ut_prev,
                vel,
                dx, dt, cT
            )

            outputs = model(u_prev,
                            ut_prev,
                            vel,
                            cT, dx * m, dt * rt, u0, dx)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)  # fine solution as target
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            u_prev, ut_prev = labels  # fine solution as input for next iteration

        if epoch % 1 == 0:
            print(datetime.datetime.now(), 'epoch %d: loss: %.5f' % (epoch + 1, np.array(loss_list).mean()))

        if epoch % 50 == 0:  # saves first models as a test
            save_model(model, model_name)
            model.to(device)

    save_model(model, model_name)





