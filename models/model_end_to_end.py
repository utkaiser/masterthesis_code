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
        self.restr_layer1 = Restr_block(in_channels_wave, 6,groups=3).double()
        self.restr_layer2 = Restr_block(6, 6*2,groups=3).double()
        self.restr_layer3 = Restr_block(6*2, 6*2,groups=3).double()
        self.restr_layer4 = Restr_block(6*2, 6*4,groups=3).double()
        self.restr_layer5 = Restr_block(6*4, 6*4,groups=3,stride=2).double()
        self.restr_layer6 = Restr_block(6*4, 6*2,groups=3).double()
        self.restr_layer7 = Restr_block(6*2, 6,groups=3).double()
        self.restr_layer8 = Restr_block(6, in_channels_wave, relu=False, batch_norm=False,groups=3).double()

        ##################### enhancing net ####################

        self.jnet = model_unet.UNet(wf=1, depth=3, scale_factor=self.res_scaler).double()


    def forward(self, x):

        u_x, u_y, u_t_c, vel = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :], x[:, 3, :, :]  # b x w x h
        u, ut = WaveSol_from_EnergyComponent_tensor(u_x.to(device), u_y.to(device), u_t_c.to(device), vel.to(device),
                                                    self.f_delta_x, torch.sum(torch.sum(u_x)))

        ###### R (restriction) ######

        if self.downsampling_net:
            restr_input = torch.stack((u.to(device), ut.to(device),vel.to(device)), dim=1).to(device)
            restr_output = self.restr_layer1(restr_input)
            restr_output = self.restr_layer2(restr_output)
            restr_output = self.restr_layer3(restr_output)
            #skip_all = phys_output #TODO: change this
            restr_output = self.restr_layer4(restr_output)
            restr_output = self.restr_layer5(restr_output)  # stride
            restr_output = self.restr_layer6(restr_output)
            restr_output = self.restr_layer7(restr_output)
            restr_output = self.restr_layer8(restr_output)

        else:
            restr_output = torch.zeros([u.shape[0], 3, 64, 64])
            restr_output[:,0,:,:] = F.upsample(u[:,:,:].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
            restr_output[:,1, :, :] = F.upsample(ut[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')
            restr_output[:,2, :, :] = F.upsample(vel[:, :, :].unsqueeze(dim=0), size=(64, 64), mode='bilinear')

        restr_fine_sol_u = torch.Tensor.double(restr_output[:, 0, :, :])  # b x w_c x h_c
        restr_fine_sol_ut = torch.Tensor.double(restr_output[:, 1, :, :])  # b x w_c x h_c
        vel_c = torch.Tensor.double(restr_output[:,2, :, :])  # b x w_c x h_c

        #TODO: print out before

        ###### G delta t (coarse iteration) ######
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u, restr_fine_sol_ut,
            vel_c, self.c_delta_x, self.c_delta_t,
            self.delta_t_star, number=1, boundary_c=self.boundary_c
        )  # b x w_c x h_c, b x w_c x h_c

        # TODO: print out after

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx, utcx, vel_c, self.f_delta_x)  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

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


        # res = torch.zeros([wx.shape[0], 3, 128, 128])  # b x c x h x w
        # res[:,0, :, :] = F.upsample(wx[:, :, :].unsqueeze(dim=1), size=(128, 128), mode='bilinear').squeeze()
        # res[:,1, :, :] = F.upsample(wy[:, :, :].unsqueeze(dim=1), size=(128, 128), mode='bilinear').squeeze()
        # res[:,2, :, :] = F.upsample(wtc[:, :, :].unsqueeze(dim=1), size=(128, 128), mode='bilinear').squeeze()