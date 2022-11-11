from torch import nn
import warnings
import models.model_unet as model_unet
from generate_data.wave_util import WaveEnergyField
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from generate_data.wave_propagation import velocity_verlet_tensor
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor
import torch
import matplotlib.pyplot as plt

class Restriction_nn(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsampling (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, in_channels_wave=2, in_channels_vel=1, res_scaler = 2, delta_t_star = .06, f_delta_x =2.0 / 128.0, boundary_c='periodic', batch_size = 1):
        super().__init__()
        # https://arxiv.org/abs/1412.6806 why I added stride

        # param setup
        self.delta_t_star = delta_t_star
        self.c_delta_x = f_delta_x * res_scaler
        self.c_delta_t = self.c_delta_x / 10
        self.f_delta_x = f_delta_x
        self.boundary_c = boundary_c
        self.batch_size = batch_size

        ##################### restriction nets ####################

        #TODO: automate if we want downsample two times
        self.phys_comp_restr_layer1 = Restr_block(in_channels_wave, 8) #TODO: check if same channel number as unet when skip_all adding
        self.phys_comp_restr_layer2 = Restr_block(8, 8 * 2, stride=2)
        self.phys_comp_restr_layer3 = Restr_block(8*2, 8*2)
        self.phys_comp_restr_layer4 = Restr_block(8*2, 8)
        self.phys_comp_restr_layer5 = Restr_block(8, in_channels_wave, relu=False, batch_norm=False)

        self.vel_restr_layer1 = Restr_block(in_channels_vel, in_channels_vel * 2)
        self.vel_restr_layer2 = Restr_block(in_channels_vel * 2, in_channels_vel * 4, stride=2)
        self.vel_restr_layer3 = Restr_block(in_channels_vel * 4, in_channels_vel * 4)
        self.vel_restr_layer4 = Restr_block(in_channels_vel * 4, in_channels_vel * 2)
        self.vel_restr_layer5 = Restr_block(in_channels_vel * 2, in_channels_vel, relu=False, batch_norm=False)


        ##################### enhancing net ####################

        self.jnet = model_unet.UNet(wf=1, depth=3, scale_factor=res_scaler).double()


    def forward(self, x):

        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]
        sumv = torch.sum(torch.sum(u_x))

        u, ut = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, self.f_delta_x, sumv)

        ###### R (restriction) ######

        # physical component restriction net
        phys_input = torch.stack((u, ut), dim=1)
        phys_output = self.phys_comp_restr_layer1(phys_input)
        skip_all = phys_output
        phys_output = self.phys_comp_restr_layer2(phys_output)
        phys_output = self.phys_comp_restr_layer3(phys_output)
        phys_output = self.phys_comp_restr_layer4(phys_output)
        phys_output = self.phys_comp_restr_layer5(phys_output)
        restr_fine_sol_u = phys_output[:, 0, :, :]  # b x w_c x h_c
        restr_fine_sol_ut = phys_output[:, 1, :, :]  # b x w_c x h_c

        # velocity component restriciton net
        vel = vel.unsqueeze(dim=1)
        vel_output = self.vel_restr_layer1(vel)
        vel_output = self.vel_restr_layer2(vel_output)
        vel_output = self.vel_restr_layer3(vel_output)
        vel_output = self.vel_restr_layer4(vel_output)
        vel_output = self.vel_restr_layer5(vel_output)
        vel_c = vel_output[:,0, :, :]  # b x w_c x h_c

        ###### G delta t (coarse iteration) ######
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, self.c_delta_x, self.c_delta_t, self.delta_t_star, number=1, boundary_c=self.boundary_c
        )# b x w_c x h_c, b x w_c x h_c

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx, utcx, vel_c,
                                                      self.f_delta_x)  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        # create input for nn
        inputs = torch.stack((wx, wy, wtc, vel_c), dim=1)  # b x 4 x 64 x 64

        ###### upsampling through nn ######
        outputs = self.jnet(inputs, skip_all=skip_all)  # b x 3 x w x h

        # change output back to wave components
        vx = outputs[:, 0, :, :]
        vy = outputs[:, 1, :, :]
        vtc = outputs[:, 2, :, :]

        # plt.imshow(WaveEnergyField(restr_fine_sol_u[0,:,:].detach().numpy(),restr_fine_sol_ut[0,:,:].detach().numpy(),vel_c[0,:,:].detach().numpy(),self.f_delta_x))
        # plt.show()

        return torch.stack((vx, vy, vtc), dim=1)


class Restr_block(nn.Module):
    def __init__(self, in_channels, out_channel, stride=1, relu=True, batch_norm=True, kernel=3, padding=1):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channel, kernel_size=kernel, padding=padding, stride = stride)
        ]
        if batch_norm: layers += [nn.BatchNorm2d(out_channel)]
        if relu: layers += [nn.ReLU()]

        self.restr = nn.Sequential(*layers)

    def forward(self, x):
        return self.restr(x)






        #resizing
        # restr_output = torch.zeros([u.shape[0], 3, 64,64])
        # restr_output[:,0,:,:] = F.upsample(u[:,:,:].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
        # restr_output[:,1, :, :] = F.upsample(ut[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
        # restr_output[:,2, :, :] = F.upsample(vel[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
        #
        # restr_fine_sol_u = torch.Tensor.double(restr_output[:, 0, :, :]) # b x w_c x h_c
        # restr_fine_sol_ut = torch.Tensor.double(restr_output[:, 1, :, :]) # b x w_c x h_c
        # vel_c = torch.Tensor.double(restr_output[:,2, :, :]) # b x w_c x h_c
        # dx = 2.0 / 128.0
        # plt.imshow(
        #     wave_util.WaveEnergyField_tensor(restr_fine_sol_u[0, :, :].detach(), restr_fine_sol_ut[0, :, :].detach(),
        #                                      vel_c[0, :, :].detach(), dx) * dx * dx)
        # plt.show()
        # combined_data = torch.concat([restr_fine_sol_u[0,:,:], F.upsample(u[:,:,:].unsqueeze(dim=0), size=(64, 64), mode='bilinear')[0,0,:,:]])
        # _min, _max = torch.min(combined_data), torch.max(combined_data)
        # fig = plt.figure(figsize=(20, 10))
        # ax1 = fig.add_subplot(1,2,1)
        # pos1 = ax1.imshow(restr_fine_sol_u.detach().numpy()[0,:,:],vmin = _min, vmax = _max)
        # plt.colorbar(pos1)
        # ax2 = fig.add_subplot(1, 2, 2)
        # pos2 = ax2.imshow(F.upsample(u[:,:,:].unsqueeze(dim=0), size=(64, 64), mode='bilinear')[0,0,:,:],vmin = _min, vmax = _max)
        # plt.colorbar(pos2)
        # plt.show()

    # a,b,c = WaveEnergyComponentField_tensor(restr_fine_sol_u,restr_fine_sol_ut,vel_c, self.f_delta_x)
    # res = torch.zeros([u.shape[0], 4, 128, 128])
    # res[:,0,:,:] = F.upsample(a.unsqueeze(dim=0), size=(128, 128), mode='bilinear')
    # res[:,1, :, :] = F.upsample(b.unsqueeze(dim=0), size=(128, 128), mode='bilinear')
    # res[:,2, :, :] = F.upsample(c.unsqueeze(dim=0), size=(128, 128), mode='bilinear')
    # res[:,3, :, :] = F.upsample(vel_c.unsqueeze(dim=0), size=(128, 128), mode='bilinear')

        # # change to energy components
        # wx, wy, wtc = WaveEnergyComponentField_tensor(ucx, utcx, vel_c,
        #                                               self.f_delta_x)  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c
        # nwx, nwy, nwtc = F.upsample(wx.unsqueeze(dim=0), size=(128, 128), mode='bilinear'), F.upsample(
        #     wy.unsqueeze(dim=0), size=(128, 128), mode='bilinear'), F.upsample(wtc.unsqueeze(dim=0), size=(128, 128),
        #                                                                        mode='bilinear')
        # # create input for nn
        # nwx, nwy, nwtc = nwx.squeeze(), nwy.squeeze(), nwtc.squeeze()
        # inputs = torch.stack((nwx, nwy, nwtc), dim=1)  # b x 4 x 64 x 64
        #
        # # ###### upsampling through nn ######
        # # outputs = self.jnet(inputs, skip_all=skip_all)  # b x 3 x w x h
        # #
        # # #change output back to wave components
        # # vx = outputs[:, 0, :, :]
        # # vy = outputs[:, 1, :, :]
        # # vtc = outputs[:, 2, :, :]