from torch import nn
import warnings
warnings.filterwarnings("ignore")
import unet
from generate_data.wave2 import velocity_verlet_time_integrator as propagate

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


    def forward(self, prev_fine_sol_u, restr_fine_sol_ut, vel, delta_t_star, dx_times_m, dt_times_rt):
        #TODO: find code for u and ut
        restr_fine_sol_u, restr_fine_sol_ut  = self.restriction(prev_fine_sol_u, restr_fine_sol_ut)

        #TODO: coarse solver
        x = propagate(restr_fine_sol_u, restr_fine_sol_ut, dx_times_m,dt_times_rt,delta_t_star)

        return self.jnet(x, vel)


