import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from generate_data.wave_propagation import velocity_verlet_tensor
from generate_data.utils_wave import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import torch


class Numerical_solver(torch.nn.Module):
    def __init__(self, boundary_condition,c_delta_x,c_delta_t,f_delta_x,delta_t_star):
        super(Numerical_solver, self).__init__()
        self.boundary_condition = boundary_condition
        self.c_delta_x = c_delta_x
        self.c_delta_t = c_delta_t
        self.f_delta_x = f_delta_x
        self.delta_t_star = delta_t_star

    def forward(self, restr_output):

        vel_c = torch.Tensor.double(restr_output[:, 3, :, :])  # b x w_c x h_c

        restr_fine_sol_u, restr_fine_sol_ut = WaveSol_from_EnergyComponent_tensor(
            torch.Tensor.double(restr_output[:, 0, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 1, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 2, :, :]).to(device),
            vel_c.to(device),
            self.c_delta_x, torch.sum(torch.sum(torch.Tensor.double(restr_output[:, 0, :, :])))
        )

        # G delta t (coarse iteration)
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u.to(device), restr_fine_sol_ut.to(device),
            vel_c.to(device), self.c_delta_x, self.c_delta_t,
            self.delta_t_star, number=1, boundary_c=self.boundary_condition
        )  # b x w_c x h_c, b x w_c x h_c

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(ucx.to(device), utcx.to(device), vel_c.to(device), self.c_delta_x)  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        # create input for nn
        return torch.stack((wx.to(device), wy.to(device), wtc.to(device), vel_c.to(device)), dim=1).to(device)  # b x 4 x 64 x 64
