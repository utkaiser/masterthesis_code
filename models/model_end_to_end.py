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
from models.restriction_models import CNN_restriction, Simple_restriction, Interpolation_net


class Restriction_nn(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsampling (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, in_channels_wave=4, param_dict=None):
        super().__init__()
        # https://arxiv.org/abs/1412.6806 why I added stride

        self.param_dict = param_dict
        self.restriction = choose_restriction(param_dict["restriction_type"])
        self.jnet = model_unet.UNet(wf=1, depth=3, scale_factor=self.param_dict["res_scaler"]).double()


    def forward(self, x):

        ###### R (restriction) ######
        restr_output, skip_all = self.restriction(x)
        vel_c = torch.Tensor.double(restr_output[:, 3, :, :])  # b x w_c x h_c
        restr_fine_sol_u, restr_fine_sol_ut = WaveSol_from_EnergyComponent_tensor(
            torch.Tensor.double(restr_output[:, 0, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 1, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 2, :, :]).to(device),
            vel_c.to(device),
            self.param_dict["c_delta_x"], torch.sum(torch.sum(torch.Tensor.double(restr_output[:, 0, :, :])))
        )

        ###### G delta t (coarse iteration) ######
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u.to(device), restr_fine_sol_ut.to(device),
            vel_c.to(device), self.param_dict["c_delta_x"], self.param_dict["c_delta_t"],
            self.param_dict["delta_t_star"], number=1, boundary_c=self.param_dict["boundary_c"]
        )  # b x w_c x h_c, b x w_c x h_c

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx.to(device), utcx.to(device), vel_c.to(device), self.param_dict["c_delta_x"])  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        # create input for nn
        inputs = torch.stack((wx.to(device), wy.to(device), wtc.to(device), vel_c.to(device)), dim=1).to(device)  # b x 4 x 64 x 64

        ##### upsampling through nn ######
        outputs = self.jnet(inputs, skip_all=skip_all)  # b x 3 x w x h

        return outputs.to(device)





def choose_restriction(restriction_type):

    if restriction_type == "simple":
        return Simple_restriction(in_channels=4)
    elif restriction_type == "cnn":
        return CNN_restriction(in_channels=4)
    elif restriction_type == "interpolation":
        return Interpolation_net(sizing_factor=2)
    else:
        raise NotImplementedError("This downsampling network has not been implemented yet!")





