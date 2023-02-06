from torch import nn
import warnings
import sys
sys.path.append("..")
from models.model_upsampling import choose_upsampling
warnings.filterwarnings("ignore")
from models.model_numerical_solver import Numerical_solver
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.models_downsampling import choose_downsampling


class Model_end_to_end(nn.Module):
    '''
            x: delta t star, dx, prev fine solution
            -> downsampling (NN1) -> coarse solution propagates -> upsample (NN2)
            y: fine solution
    '''

    def __init__(self, param_dict, downsampling_type, upsampling_type, res_scaler, model_res):
        super().__init__()
        # https://arxiv.org/abs/1412.6806 why I added stride

        self.param_dict = param_dict
        self.model_downsampling = choose_downsampling(downsampling_type, res_scaler, model_res)
        self.model_downsampling.to(device)
        self.model_numerical = Numerical_solver(param_dict["boundary_c"], param_dict["c_delta_x"], param_dict["c_delta_t"],param_dict["f_delta_x"],param_dict["delta_t_star"])
        self.model_numerical.to(device)
        self.model_upsampling = choose_upsampling(upsampling_type, res_scaler)
        self.model_upsampling.to(device)


    def forward(self, x):

        ###### R (restriction) ######
        downsampling_res, skip_all = self.model_downsampling(x)

        # velocity verlet
        prop_result = self.model_numerical(downsampling_res)

        ##### upsampling through nn ######
        outputs = self.model_upsampling(prop_result.to(device), skip_all=skip_all)  # b x 3 x w x h

        return outputs.to(device)



