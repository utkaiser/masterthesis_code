import numpy as np
import torch

import WaveUtil
import OPPmodel


# Evaluate NN on Solution
def ApplyNet2WaveSol(w0,wt0,w,wt,c,dx,net):
    
    w0x,w0y,wt0c = WaveUtil.WaveEnergyComponentField(np.expand_dims(w0,axis=2),np.expand_dims(wt0,axis=2),c,dx)
    wx,wy,wtc = WaveUtil.WaveEnergyComponentField(np.expand_dims(w,axis=2),np.expand_dims(wt,axis=2),c,dx)
    
    w0x = torch.from_numpy(np.transpose(w0x,(2,0,1)))
    w0y = torch.from_numpy(np.transpose(w0y,(2,0,1)))
    wt0c = torch.from_numpy(np.transpose(wt0c,(2,0,1)))
    wx = torch.from_numpy(np.transpose(wx,(2,0,1)))
    wy = torch.from_numpy(np.transpose(wy,(2,0,1)))
    wtc = torch.from_numpy(np.transpose(wtc,(2,0,1)))
    
    inputs = torch.stack((w0x,w0y,wt0c,wx,wy,wtc),dim=1)
    
    outputs = net(inputs)
    
    vx = outputs[0,0,:,:]
    vy = outputs[0,1,:,:]
    vtc = outputs[0,2,:,:]
    
    sumv = np.sum(w0)
    u,ut = WaveUtil.WaveSol_from_EnergyComponent(vx.detach().numpy(),vy.detach().numpy(),\
                                                 vtc.detach().numpy(),c,dx,sumv)
    
    return u,ut


# Evaluate Procrustes model on Solution
def ApplyOPP2WaveSol(w,wt,c,dx,opmap):
    P,_,Q = opmap
    
    wx,wy,wtc = WaveUtil.WaveEnergyComponentField(np.expand_dims(w,axis=2),np.expand_dims(wt,axis=2),c,dx)
    
    vx,vy,vtc = OPPmodel.ProcrustesShift(P,Q,(wx,wy,wtc),datmode='numpy')
    
    sumv = np.sum(w)
    
    u,ut = WaveUtil.WaveSol_from_EnergyComponent(np.squeeze(vx),np.squeeze(vy),\
                                                 np.squeeze(vtc),c,dx,sumv)
    return u,ut