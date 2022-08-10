import sys
import time
import numpy as np

import torch
#import torchvision.transforms as transforms
import torch.nn as nn
import unet_Linear as unetLin
#import cblendunet as cnet
import torch.optim as optim


def npdat2Tensor(nda):
    ndt = np.transpose(nda,(2,0,1))
    ndt = torch.from_numpy(ndt)

    return ndt


if __name__ == "__main__":
    """
    train neural network model from pair data coarse and fine solutions
    """
    batchsize = 128
    learning_rate = 1e-4
    epochs = 850
    nlayer = 3
    netmod = 'NL'
    datasetno = [22,]
    wf = 1
    fine_coarse_scale = 4
    print(netmod)
    
    # Activation function is either identity (linear) or ReLU
    if netmod == 'Linear':
        net = unetLin.UNet(wf=wf,depth=nlayer,acti_func='identity')
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif netmod =='NL':
        net = unetLin.UNet(wf=wf,depth=nlayer,acti_func='relu', finescale_factor=fine_coarse_scale)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #elif netmod=='ComNet' or netmod=='ComNetNL':
    #    net = cnet.CBlend_roll(syn_depth=nlayer,syn_wf=1,exp_depth=6,exp_wf=1,exp1_acti='identity',exp2_acti='relu',net1dir='LinearModule_6layer_10Kdata[4, 5].pt',net2dir='NLModule_6layer_10Kdata[4, 5].pt')
    #    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net = net.double()
    
    # Continue training the network 
    # net.load_state_dict(torch.load('./NLModule_w1_3layer_data[22].pt'))
    # net.eval()
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters ',params)
    
    criterion = nn.MSELoss() # loss function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(device)
    print('Number of layers ',nlayer)
    
    print('Data number ',datasetno)

    start_time = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        training_loss = 0.0
        id_loss = 0.0
        
        for i in datasetno:
            npz_PropS = np.load('./data/traindata_name'+str(i)+'.npz')
            datasize = npz_PropS['vel'].shape[2]
    
            inputdata = torch.stack((npdat2Tensor(npz_PropS['Ucx']),\
                                     npdat2Tensor(npz_PropS['Ucy']),\
                                     npdat2Tensor(npz_PropS['Utc']),\
                                     npdat2Tensor(npz_PropS['vel'])\
    				 ),dim=1)
                
    
            outputdata = torch.stack((npdat2Tensor(npz_PropS['Ufx']),\
                                      npdat2Tensor(npz_PropS['Ufy']),\
                                      npdat2Tensor(npz_PropS['Utf'])),dim=1)
        
    
            wavedataset = torch.utils.data.TensorDataset(inputdata,outputdata)
            #wavedataset = torch.utils.data.Subset(wavedataset,list(range(5000)))
            trainLoader = torch.utils.data.DataLoader(wavedataset, batch_size=batchsize,shuffle=True,num_workers=1)   
            
            for i, data in enumerate(trainLoader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                training_loss += loss.item()
                
                id_loss += criterion(nn.functional.upsample(inputs[:,:3,:,:],scale_factor=fine_coarse_scale,mode='bilinear'),labels).item()
    
            if epoch % 1 == 0:    # print every epochs
                    print('[%d] training data loss: %.5f | coarse loss: %.5f ' %
                          (epoch + 1, training_loss, id_loss))
                                        
    
    end_time = time.time()
    print('Training done ',(end_time-start_time))
    
    device = torch.device("cpu") 
    net.to(device)   
    torch.save(net.state_dict(),netmod+'Module_w'+str(wf)+'_'+str(nlayer)+'layer_data'+str(datasetno)+'.pt')                            
