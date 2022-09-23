import numpy as np
import torch
import WaveUtil as WaveUtil
import OPPmodel as OPPmodel
from skimage.transform import resize

def ApplyNet2WaveSol(w0,wt0,w,wt,c,dx,net):
    # Evaluate NN on Solution
    
    wx,wy,wtc = WaveUtil.WaveEnergyComponentField(np.expand_dims(w, axis=2), np.expand_dims(wt, axis=2), resize(c, [64, 64], order=4), dx)

    wx = torch.from_numpy(np.transpose(wx,(2,0,1)))
    wy = torch.from_numpy(np.transpose(wy,(2,0,1)))
    wtc = torch.from_numpy(np.transpose(wtc,(2,0,1)))

    c_tmp = np.expand_dims(resize(c, [64, 64], order=4),2)
    c_tmp = torch.from_numpy(np.transpose(c_tmp,(2,0,1)))
    #inputs = torch.stack((w0x,w0y,wt0c,wx,wy,wtc),dim=1)
    inputs = torch.stack((wx, wy, wtc, c_tmp), dim=1)

    outputs = net(inputs)
    
    vx = outputs[0,0,:,:]
    vy = outputs[0,1,:,:]
    vtc = outputs[0,2,:,:]
    
    sumv = np.sum(w0)

    u,ut = WaveUtil.WaveSol_from_EnergyComponent(vx.detach().numpy(), vy.detach().numpy(),
                                                 vtc.detach().numpy(), c, dx, sumv)

    #u, ut = resize(u, [128,128], order=4), resize(ut, [128,128], order=4)
    return u, ut



def ApplyOPP2WaveSol(w,wt,c,dx,opmap):
    # Evaluate Procrustes model on Solution

    P,_,Q = opmap
    wx,wy,wtc = WaveUtil.WaveEnergyComponentField(np.expand_dims(w, axis=2), np.expand_dims(wt, axis=2), c, dx)
    vx,vy,vtc = OPPmodel.ProcrustesShift(P, Q, (wx, wy, wtc), datmode='numpy')
    sumv = np.sum(w)
    u,ut = WaveUtil.WaveSol_from_EnergyComponent(np.squeeze(vx), np.squeeze(vy),
                                                 np.squeeze(vtc), c, dx, sumv)
    return u, ut