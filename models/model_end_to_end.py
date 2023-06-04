import sys

from models.utils import choose_optimizer, choose_loss_function

sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")
import logging
from torch import nn, save
from models.model_upsampling import choose_upsampling
from models.model_numerical_solver import Numerical_solver
from models.models_downsampling import choose_downsampling
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from os import path


class Model_end_to_end(
    nn.Module
):
    '''
    main end-to-end module class that builds interaction of different components
    down sampling + coarse solver + up sampling
    '''

    def __init__(
            self,
            param_dict,
            downsampling_type,
            upsampling_type,
            res_scaler,
            model_res
    ):
        '''
        Parameters
        ----------
        param_dict : (dict) contains parameters to set up model
        downsampling_type : (string) defines down sampling component
        upsampling_type : (string) defines up sampling component
        res_scaler : (int) down scaling factor of input, usually 2 or 4
        model_res : (int) resolution model can handle
        '''

        super().__init__()

        self.param_dict = param_dict
        self.model_downsampling = choose_downsampling(downsampling_type, res_scaler, model_res)
        self.model_downsampling.to(device)
        self.model_numerical = Numerical_solver(param_dict["boundary_c"], param_dict["c_delta_x"], param_dict["c_delta_t"],param_dict["delta_t_star"])
        self.model_numerical.to(device)
        self.model_upsampling = choose_upsampling(upsampling_type, res_scaler)
        self.model_upsampling.to(device)


    def forward(
            self,
            x
    ):
        '''
        Parameters
        ----------
        x : (pytorch tensor) input x as defined in paper with three wave energy components and velocity profile

        Returns
        -------
        propagates waves one time step delta_t_star using end-to-end model
        '''

        # restriction component
        downsampling_res, _ = self.model_downsampling(x)  # second component is skip connection

        # velocity verlet
        prop_result = self.model_numerical(downsampling_res)

        # up sampling component
        outputs = self.model_upsampling(prop_result.to(device), skip_all=None)

        return outputs.to(device)


def get_model(
        param_dict,
        res_scaler,
        model_res,
        down_sampling_component = "Interpolation",
        up_sampling_component = "UNet3",
        model_path = '../results/run_3/good/saved_model_Interpolation_UNet3_AdamW_SmoothL1Loss_2_128_False_15.pt'
):
    '''
    Parameters
    ----------
    param_dict : (dict) contains parameters to set up model
    res_scaler : (int) down scaling factor of input, usually 2 or 4
    model_res : (int) resolution model can handle
    down_sampling_component: (string) choice of down sampling component
    up_sampling_component: (string)  choice of up sampling component
    model_path : (string) path to model parameters saved in ".pt"-file

    Returns
    -------
    load pre-trained model and retunr model
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(" ".join(["gpu available:", str(torch.cuda.is_available()), "| n of gpus:", str(torch.cuda.device_count())]))
    model = Model_end_to_end(param_dict, down_sampling_component, up_sampling_component, res_scaler, model_res).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model.load_state_dict(torch.load(model_path))

    return model


def save_model(
        model,
        model_name,
        dir_path='results/run_3/'
):
    '''
    Parameters
    ----------
    model : end-to-end model instance
    model_name : name of end-to-end model
    dir_path : directory where to save model in

    Returns
    -------
    save {model} as ".pt"-file
    '''

    model.to(torch.device("cpu"))
    saving_path = dir_path + 'saved_model_' + model_name + '.pt'
    if not path.isfile(saving_path):
        return save(model.state_dict(), saving_path)
    else:
        raise MemoryError("File (.pt) already exists.")



def setup_model(param_d, downsampling_model, upsampling_model,model_res, lr,weight_decay, device, weighted_loss):

    # model setup
    model = Model_end_to_end(param_d, downsampling_model, upsampling_model, param_d["res_scaler"], model_res).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    optimizer = choose_optimizer(param_d["optimizer_name"], model, lr,weight_decay)
    loss_f = choose_loss_function(param_d["loss_function_name"])

    if weighted_loss == False:
        label_distr_shift = None
    else:
        label_distr_shift = 1

    return model, optimizer, loss_f, label_distr_shift