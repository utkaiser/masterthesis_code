import numpy as np
import torch
from skimage.transform import resize
import wave_util
import opp_model

def ApplyJNet2WaveSol(w,wt,c,dx,net,m=2):
    ''' 
    Apply network net to the given wavefield.
    The wavefield is transformed into wave energy component, on which
    the network applies. Then the network output is transformed back
    to wavefield.
    Intput
    w,wt: wavefield on coarse grid size
    c: wavespeed on fine grid
    dx: fine grid size
    net: JNet
    Output
    u,ut: processed wavefield on fine grid size
    '''
    
    c_coarse = resize(c,w.shape,order=4)
    wx,wy,wtc = wave_util.WaveEnergyComponentField(np.expand_dims(w,axis=2),np.expand_dims(wt,axis=2),c_coarse,dx*m)

    wx = torch.from_numpy(np.transpose(wx,(2,0,1)))
    wy = torch.from_numpy(np.transpose(wy,(2,0,1)))
    wtc = torch.from_numpy(np.transpose(wtc,(2,0,1)))

    ctensor = torch.from_numpy(np.transpose(np.expand_dims(c_coarse,axis=2),(2,0,1)))
    inputs = torch.stack((wx,wy,wtc,ctensor),dim=1)
    
    outputs = net(inputs)
    
    vx = outputs[0,0,:,:]
    vy = outputs[0,1,:,:]
    vtc = outputs[0,2,:,:]
    
    sumv = np.sum(resize(w,c.shape,order=4))
    u,ut = wave_util.WaveSol_from_EnergyComponent(vx.detach().numpy(),vy.detach().numpy(),
                                                 vtc.detach().numpy(),c,dx,sumv)
    
    return u,ut


# Evaluate Procrustes model on Solution
def ApplyOPP2WaveSol(w,wt,c,dx,opmap):
    '''
    Apply Procrustes matrix-operator model to wavefield.
    The wavefield is transformed into wave energy component, to which
    the model multiplies. Then the output is transformed back
    to wavefield.
    Intput
    w,wt: wavefield on coarse grid size
    c: wavespeed on fine grid
    dx: fine grid size
    opmap: low-rank matrices P,Q
    Output
    u,ut: processed wavefield on fine grid size
    '''
    P,_,Q = opmap
    
    wx,wy,wtc = wave_util.WaveEnergyComponentField(np.expand_dims(w,axis=2),np.expand_dims(wt,axis=2),c,dx)
    
    vx,vy,vtc = opp_model.ProcrustesShift(P,Q,(wx,wy,wtc),datmode='numpy')

    sumv = np.sum(w)
    
    u,ut = wave_util.WaveSol_from_EnergyComponent(np.squeeze(vx),np.squeeze(vy),
                                                 np.squeeze(vtc),c,dx,sumv)
    return u,ut