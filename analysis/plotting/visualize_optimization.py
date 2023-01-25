import sys
sys.path.append("..")
from models.model_end_to_end import Model_end_to_end
from models.optimization.utils_optimization import get_solver_solution, smaller_crop
from generate_data.generate_test_environment import get_velocity_crop
from analysis.plotting.plot_wavefield import get_ticks_fine, plot_wavefield_optimization
from analysis.plotting.plot_heatmap import plot_heatmap_optimization
from generate_data.initial_conditions import init_gaussian_parareal
from generate_data.utils_wave import crop_center
import torch
import numpy as np
from models.model_utils import get_params
from models.optimization.parallel_scheme import parareal_scheme
from models.optimization.parallel_procrustes_scheme import parareal_procrustes_scheme


def vis_parareal(vel_name, big_vel, folder_name, n_snaps, mode="parareal"):

    # data
    vel = crop_center(big_vel,128,128)
    u_0 = torch.concat([init_gaussian_parareal(256,big_vel), big_vel.unsqueeze(dim=0).unsqueeze(dim=0)],dim=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_end_to_end(param_dict=get_params("0")).double().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10_2.pt'))
    model.eval()

    with torch.no_grad():
        parareal_tensor = torch.zeros([5,])# choose_optimization_method(model, u_0,big_vel,n_snaps,mode=mode)  # k x s x b x c x w x h  # TODO: decide if parareal or parareal + procrustes
        coarse_solver_tensor = get_solver_solution(smaller_crop(u_0[:, :3, :, :]), 11,smaller_crop(u_0[:, 3, :,:]).unsqueeze(dim=0), solver="coarse")  # s x b x c x w x h
        fine_solver_tensor = get_solver_solution(u_0[:, :3, :, :], 11,u_0[:, 3, :, :].unsqueeze(dim=0), solver="fine")  # s x b x c x w x h
        ticks = get_ticks_fine(fine_solver_tensor, vel)  # s x 3

        plot_wavefield_optimization(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, vel, vel_name, folder_name)
        plot_heatmap_optimization(fine_solver_tensor, parareal_tensor, vel, vel_name, folder_name)

        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_coarse.npy', coarse_solver_tensor.numpy())
        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_fine.npy', fine_solver_tensor.numpy())
        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_parareal.npy', parareal_tensor.numpy())


def choose_optimization_method(model,u_0,n_snaps,mode="parareal"):
    if mode == "parareal":
        return parareal_scheme(model, u_0, n_snaps)
    elif mode == "parareal_procrustes":
        return parareal_procrustes_scheme(model, u_0, n_snaps)
    else:
        raise NotImplementedError("This optimization method has not been implemented yet!")


def vis_multiple_init_cond():
    velocities = {
        "diagonal": torch.from_numpy(get_velocity_crop(256, "diagonal")),
        # "marmousi": torch.from_numpy(get_velocity_crop(256, "marmousi")),
        # "marmousi2": torch.from_numpy(get_velocity_crop(256, "marmousi2")),
        # "bp": torch.from_numpy(get_velocity_crop(256, "bp")),
        # "three_layers": torch.from_numpy(get_velocity_crop(256, "three_layers")),
        # "crack_profile": torch.from_numpy(get_velocity_crop(256, "crack_profile")),
        # "high_frequency": torch.from_numpy(get_velocity_crop(256, "high_frequency"))
    }

    for vel_name, vel in velocities.items():
        print("-"*20,vel_name,"-"*20)
        vis_parareal(vel_name, vel, folder_name = "procrustes", mode="parareal", n_snaps=11)


if __name__ == '__main__':
    vis_multiple_init_cond()