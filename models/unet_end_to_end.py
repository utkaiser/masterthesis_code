from torch import nn
import warnings
warnings.filterwarnings("ignore")
import unet_old as unet
from generate_data.wave2 import velocity_verlet_tensor as propagate
from generate_data.WaveUtil import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent
import numpy as np
import torch
from models.model_utils import npdat2Tensor_tensor, npdat2Tensor

class restriction_nn(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsamply (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, in_channels=3, n_classes=2, down_factor = 2):
        super(restriction_nn, self).__init__()

        # params
        self.delta_t_star = .2
        self.dx = 2.0 / 128.0
        self.dx_times_m = self.dx * 2
        self.dt_times_rt = self.dx_times_m / 10

        # restriction nn
        self.restriction = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, padding=1, stride=down_factor),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, 3, kernel_size=1, bias=False)
        ])

        # enhancing nn
        self.jnet = unet.UNet(wf=1, depth=3, scale_factor=down_factor)


    def forward(self, x):

        #R (restriction)
        restr_output = self.restriction(x)


        restr_fine_sol_u, restr_fine_sol_ut, vel_c = restr_output[:,0,:,:][0], restr_output[:,1,:,:][0], restr_output[:,2,:,:][0]

        #G delta t (coarse iteration)
        ucx, utcx = propagate(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, self.dx_times_m, self.dt_times_rt, self.delta_t_star
        )

        #change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(torch.unsqueeze(ucx,2),
                                               torch.unsqueeze(utcx,2),
                                               vel_c, self.dx)

        #create input for nn
        inputs = torch.stack((npdat2Tensor_tensor(wx),
                              npdat2Tensor_tensor(wy),
                              npdat2Tensor_tensor(wtc),
                              npdat2Tensor_tensor(torch.unsqueeze(vel_c, 2))), dim=1)

        #upsampling through nn
        outputs = self.jnet(inputs)

        #change output back to wave components
        vx = outputs[0, 0, :, :]
        vy = outputs[0, 1, :, :]
        vtc = outputs[0, 2, :, :]
        sumv = torch.sum(torch.sum(x[:,0,:,:][0]))
        u, ut = WaveSol_from_EnergyComponent(vx.detach().numpy(), vy.detach().numpy(),
                                             vtc.detach().numpy(), x[:,2,:,:].detach().numpy()[0], self.dx, sumv)

        u,ut = torch.unsqueeze(u,2), torch.unsqueeze(ut,2)
        return torch.stack((npdat2Tensor_tensor(u), npdat2Tensor_tensor(ut)), dim=1)


