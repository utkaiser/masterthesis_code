from analysis.utils_analysis import get_solver_solution, one_iteration_pseudo_spectral
from analysis.visualize_results.visualize_training import visualize_wavefield
import random
import torch
from generate_data.initial_conditions import get_velocities, initial_condition_gaussian, get_velocity_crop
from generate_data.utils_wave import WaveEnergyComponentField_end_to_end, WaveSol_from_EnergyComponent_tensor, \
    WaveEnergyComponentField_tensor, WaveEnergyField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor
from models.model_utils import sample_label_normal_dist, get_wavefield, round_loss, compute_loss, compute_loss2
from generate_data.utils_wave import WaveSol_from_EnergyComponent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.optimization_models.parallel_scheme import parareal_scheme, one_iteration_pseudo_spectral_tensor, \
    parareal_scheme2
import matplotlib.pyplot as plt
import numpy as np



def model_optimization_solution(data, model, loss_f, label_distr_shift, mode, i, epoch,
                                vis_path, vis_save, optimization_type, multi_step, res):

    batch_size, n_snaps, c, w, h = data.shape
    loss_list = []
    if mode == "train":
        # random.choices(range(n_snaps - 2), k=n_snaps)
        vel = torch.from_numpy(get_velocity_crop(256, 1, "diagonal", "periodic", "parareal", "../")).squeeze()
        u_0 = torch.concat([initial_condition_gaussian(vel,128,"absorbing", 256,"parareal","parareal"), vel.unsqueeze(dim=0).unsqueeze(dim=0)], dim=1).squeeze()  # c x w x h
        u_n_k = u_0[:3].clone()
        fine_sol = torch.zeros([1, 8, 3, 256, 256])

        for s in range(8):
            fine_sol[0, s] = u_n_k
            u_n_k = torch.concat([u_n_k, vel.unsqueeze(dim=0)], dim=0).unsqueeze(dim=0)
            u_n_k = one_iteration_pseudo_spectral(u_n_k).squeeze()

        for input_idx in [0,0]:
            label_range = 5  # sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, multi_step)
            if optimization_type == "parareal":
                loss_list += parareal_scheme2(model, input_idx, 2, label_range, loss_f, fine_sol, vel)
                # loss_list += parareal_scheme(model, u_0.unsqueeze(dim=0), fine_sol)

            else:  # optimization_type == "procrustes"
                pass

    else:  # validate

        visualize_list = []
        input_tensor = data[:, 0].clone()  # b x 4 x w x h
        vel = input_tensor[:, 3].unsqueeze(dim=1)

        for label_idx in range(1, n_snaps):
            label = data[:, label_idx, :3]  # b x 3 x w x h
            output = model(input_tensor)
            val_loss = loss_f(output, label)
            loss_list.append(val_loss.item())

            if i == 0:
                # save only first element of batch
                visualize_list.append((val_loss.item(), output[0].detach().cpu(),
                                       label[0].detach().cpu()))
            input_tensor = torch.cat((output, vel), dim=1)

        if i == 0:
            visualize_wavefield(epoch, visualize_list, input_tensor[0, 3].cpu(), vis_save=vis_save,
                                vis_path=vis_path, initial_u=data[:, 0])

    return loss_list