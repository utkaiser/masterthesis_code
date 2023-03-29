import sys
sys.path.append("..")
import logging
from torch import nn
import warnings
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
        downsampling_res, _ = self.model_downsampling(x)

        # velocity verlet
        prop_result = self.model_numerical(downsampling_res)

        ##### upsampling through nn ######
        outputs = self.model_upsampling(prop_result.to(device), skip_all=None)  # b x 3 x w x h

        return outputs.to(device)


def get_model(param_dict, res_scaler, model_res):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(" ".join(["gpu available:", str(torch.cuda.is_available()), "| n of gpus:", str(torch.cuda.device_count())]))
    model = Model_end_to_end(param_dict, "Interpolation", "UNet3", res_scaler, model_res).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model.load_state_dict(torch.load('../../results/run_3/good/saved_model_Interpolation_UNet3_AdamW_SmoothL1Loss_2_128_False_15.pt'))

    return model

