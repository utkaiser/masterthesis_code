from torch import nn
import warnings
import models.model_unet as model_unet
warnings.filterwarnings("ignore")
from generate_data.wave_propagation import velocity_verlet_tensor
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor
import torch

class restriction_nn(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsampling (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, in_channels=3, n_classes=3, res_scaler = 2, delta_t_star = .2, f_delta_x = 2.0/128.0):
        super(restriction_nn, self).__init__()

        # param setup
        self.delta_t_star = delta_t_star
        self.c_delta_x = f_delta_x * res_scaler
        self.c_delta_t = self.c_delta_x / 10
        self.f_delta_x = f_delta_x


        ##################### restriction net ####################

        downsample_layers = [
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU()
        ]

        for i in range(res_scaler // 2):
            downsample_layers += [
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),  # reduces resolution, or could use maxpooling
                nn.BatchNorm2d(in_channels * 4),
                nn.ReLU()
            ]

        downsample_layers += [
            nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, 3, kernel_size=3, padding=1)
        ]

        self.restriction_net = nn.Sequential(*downsample_layers)


        ##################### enhancing net ####################

        self.jnet = model_unet.UNet(wf=1, depth=n_classes, scale_factor=res_scaler).double()


    def forward(self, x):

        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]
        sumv = torch.sum(torch.sum(u_x))

        u, ut = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, self.f_delta_x, sumv)
        phys_input = torch.stack((u, ut, vel), dim=1)

        # R (restriction)
        restr_output = self.restriction_net(phys_input) # b x 3 x w_c x h_c

        restr_fine_sol_u = restr_output[:, 0, :, :] # b x w_c x h_c
        restr_fine_sol_ut = restr_output[:, 1, :, :] # b x w_c x h_c
        vel_c = restr_output[:,2, :, :] # b x w_c x h_c

        # G delta t (coarse iteration); checked, this is valid for batching
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, self.c_delta_x, self.c_delta_t, self.delta_t_star, number=1
        ) # b x w_c x h_c, b x w_c x h_c

        #change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx,utcx,vel_c, self.f_delta_x) # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        #create input for nn
        inputs = torch.stack((wx, wy, wtc, vel_c), dim=1) # b x 4 x 64 x 64

        #upsampling through nn
        outputs = self.jnet(inputs)  # b x 3 x w x h

        #change output back to wave components
        vx = outputs[:, 0, :, :]
        vy = outputs[:, 1, :, :]
        vtc = outputs[:, 2, :, :]

        return torch.stack((vx, vy, vtc), dim=1)


