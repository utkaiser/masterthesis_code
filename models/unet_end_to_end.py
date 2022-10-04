from torch import nn
import warnings
warnings.filterwarnings("ignore")
import unet
from generate_data.wave2 import velocity_verlet_time_integrator as propagate
from generate_data.WaveUtil import WaveEnergyComponentField, WaveSol_from_EnergyComponent
import numpy as np
import torch
from models.model_utils import npdat2Tensor

class restriction_nn(nn.Module):

    def __init__(self, in_channels=3, n_classes=3, down_factor = 2):
        super(restriction_nn, self).__init__()

        self.restriction = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels*2, in_channels*4, kernel_size=3, bias=True, padding=1, stride=down_factor),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(),
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, n_classes, kernel_size=1, bias=False)
        ])

        self.jnet = unet.UNet(wf=1, depth=3, scale_factor=down_factor)


    def forward(self, prev_fine_sol_u, restr_fine_sol_ut, vel, delta_t_star, dx_times_m, dt_times_rt, w0, dx):

        #R (restriction)
        restr_fine_sol_u, restr_fine_sol_ut, vel_c  = self.restriction(prev_fine_sol_u, restr_fine_sol_ut, vel)

        #G delta t (coarse iteration)
        ucx, utcx = propagate(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, dx_times_m, dt_times_rt, delta_t_star
        )

        #change to energy components
        wx, wy, wtc = WaveEnergyComponentField(np.expand_dims(ucx, axis=2),
                                               np.expand_dims(utcx, axis=2),
                                               vel_c, dx)

        #create input for nn
        inputs = torch.stack((npdat2Tensor(wx),
                              npdat2Tensor(wy),
                              npdat2Tensor(wtc),
                              vel_c), dim=1)

        #upsampling through nn
        outputs = self.jnet(inputs)

        #change output back to wave components
        vx = outputs[0, 0, :, :]
        vy = outputs[0, 1, :, :]
        vtc = outputs[0, 2, :, :]
        sumv = np.sum(w0)
        u, ut = WaveSol_from_EnergyComponent(vx.detach().numpy(), vy.detach().numpy(),
                                             vtc.detach().numpy(), vel, dx, sumv)

        return u, ut


