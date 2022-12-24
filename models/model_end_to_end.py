from torch import nn
import warnings
import sys
sys.path.append("..")
warnings.filterwarnings("ignore")
from models import model_unet
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from generate_data.wave_propagation import velocity_verlet_tensor
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from generate_data import wave_util

class Restriction_nn(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsampling (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, in_channels_wave=3, param_dict=None):
        super().__init__()
        # https://arxiv.org/abs/1412.6806 why I added stride

        # param setup
        self.delta_t_star = param_dict["delta_t_star"]
        self.res_scaler = param_dict["res_scaler"]
        self.f_delta_x = param_dict["f_delta_x"]
        self.f_delta_t = param_dict["f_delta_t"]
        self.c_delta_x = param_dict["c_delta_x"]
        self.c_delta_t = param_dict["c_delta_t"]
        self.boundary_c = param_dict["boundary_c"]
        self.downsampling_net = param_dict["downsampling_net"]

        ##################### restriction nets ####################

        #TODO: automate if we want downsample two times
        self.restr_layer1 = Restr_block(in_channels_wave, in_channels_wave*2,groups=in_channels_wave).double()
        self.restr_layer2 = Restr_block(in_channels_wave*2, in_channels_wave*4,groups=in_channels_wave).double()
        self.restr_layer3 = Restr_block(in_channels_wave*4, in_channels_wave*4,groups=in_channels_wave).double()
        self.restr_layer4 = Restr_block(in_channels_wave*4, in_channels_wave*8,groups=in_channels_wave).double()
        self.restr_layer5 = Restr_block(in_channels_wave*8, in_channels_wave*8,groups=in_channels_wave,stride=2).double()
        self.restr_layer6 = Restr_block(in_channels_wave*8, in_channels_wave*4,groups=in_channels_wave).double()
        self.restr_layer7 = Restr_block(in_channels_wave*4, in_channels_wave*2,groups=in_channels_wave).double()
        self.restr_layer8 = Restr_block(in_channels_wave*2, in_channels_wave, relu=False, batch_norm=False,groups=in_channels_wave).double()

        ##################### enhancing net ####################

        self.jnet = model_unet.UNet(wf=1, depth=3, scale_factor=self.res_scaler).double()


    def forward(self, x):

        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]  # b x w x h

        ###### R (restriction) ######

        if self.downsampling_net:
            # restr_input = torch.stack((u.to(device), ut.to(device),vel.to(device)), dim=1).to(device)
            # restr_output = self.restr_layer1(restr_input)
            # restr_output = self.restr_layer2(restr_output)
            # restr_output = self.restr_layer3(restr_output)
            # #skip_all = phys_output #TODO: change this
            # restr_output = self.restr_layer4(restr_output)
            # restr_output = self.restr_layer5(restr_output)  # stride
            # restr_output = self.restr_layer6(restr_output)
            # restr_output = self.restr_layer7(restr_output)
            # restr_output = self.restr_layer8(restr_output)
            pass

        else:
            restr_output = torch.zeros([u_x.shape[0], 4, 64, 64])
            restr_output[:,0,:,:] = F.upsample(u_x[:,:,:].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
            restr_output[:,1, :, :] = F.upsample(u_y[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
            restr_output[:,2, :, :] = F.upsample(u_t_c[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
            restr_output[:, 3, :, :] = F.upsample(vel[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')

        restr_fine_sol_u_x = torch.Tensor.double(restr_output[:, 0, :, :])  # b x w_c x h_c
        restr_fine_sol_u_y = torch.Tensor.double(restr_output[:, 1, :, :])  # b x w_c x h_c
        restr_fine_sol_u_t_c = torch.Tensor.double(restr_output[:, 2, :, :])  # b x w_c x h_c
        vel_c = torch.Tensor.double(restr_output[:, 3, :, :])  # b x w_c x h_c

        restr_fine_sol_u, restr_fine_sol_ut = WaveSol_from_EnergyComponent_tensor(restr_fine_sol_u_x.to(device), restr_fine_sol_u_y.to(device), restr_fine_sol_u_t_c.to(device), vel_c.to(device),
                                                    self.c_delta_x, torch.sum(torch.sum(restr_fine_sol_u_x)))


        ###### G delta t (coarse iteration) ######
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, self.c_delta_x, self.c_delta_t,
            self.delta_t_star, number=1, boundary_c=self.boundary_c
        )  # b x w_c x h_c, b x w_c x h_c

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx, utcx, vel_c, self.c_delta_x)  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        # create input for nn
        inputs = torch.stack((wx.to(device), wy.to(device), wtc.to(device), vel_c.to(device)), dim=1).to(device)  # b x 4 x 64 x 64

        ##### upsampling through nn ######
        outputs = self.jnet(inputs) #, skip_all=skip_all)  # b x 3 x w x h

        return outputs


class Restr_block(nn.Module):
    def __init__(self, in_channels, out_channel, stride=1, relu=True, batch_norm=True, kernel=3, padding=1, groups=1):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel, padding=padding, stride = stride, groups=groups)
        ]
        if batch_norm: layers += [nn.BatchNorm2d(out_channel)]
        if relu: layers += [nn.ReLU()]

        self.restr = nn.Sequential(*layers)

    def forward(self, x):
        return self.restr(x)
