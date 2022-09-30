import numpy as np
import generate_data.wave2 as w2
from model_utils import fetch_data
from unet_end_to_end import restriction_nn
import torch.optim as optim
import torch.nn as nn

def propagate_end_to_end():

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
    np.random.seed = 21
    center = np.array([-0.8,-0.8])

    #training params
    batch_size = 32
    lr = .01
    gamma = .991
    fine_coarse_scale = 2
    n_epochs = 500
    nlayer = 3
    wf = 1
    continue_training = False

    ### data ###
    '''
    x: delta t star, dx, prev fine solution
    -> downsamply (NN1) -> coarse solution propagates -> upsample (NN2)
    y: fine solution
    '''

    #TODO: find solution if dataloader is used

    vel = 1. + 0.0*yy - 0.5*(np.abs(yy+xx-0.)>0.4) + 0.*(np.abs(xx-0.4)<0.2)*(np.abs(yy-0.5)<0.1)
    u0 = np.exp(-250.0 * (0.2 * (xx - center[0]) ** 2 + (yy - center[1]) ** 2)) * np.cos(8 * np.pi * (yy - center[1]))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    u = np.zeros([xx.shape[0], xx.shape[1], mt])
    ut = np.zeros([xx.shape[0], xx.shape[1], mt])
    u[:, :, 0] = u0
    ut[:, :, 0] = ut0
    # train_loaders = fetch_data('../data/traindata_name14.npz',
    #                   batchsize=batch_size,
    #                   shuffle=True)


    ### models ###

    model = restriction_nn()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.SmoothL1Loss()  # before: MSE loss
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)





    for j in range(1,mt):
        uc,utc = w2.velocity_verlet_time_integrator(resize(u[:,:,j-1],[ny,nx],order=4),
                             resize(ut[:,:,j-1],[ny,nx],order=4),
                             resize(vel,[ny,nx],order=4),dx*m,dt*rt,cT)




        u[:,:,j],utnn2[:,:,j] = wp.ApplyNet2WaveSol(u0,ut0,uc,utc,vel,dx,tir_model)











