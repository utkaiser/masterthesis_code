import sys
from analysis.utils_analysis import get_ticks_fine
sys.path.append("..")
from generate_data.initial_conditions import initial_condition_gaussian, get_velocity_crop, get_velocity_dict
from models.model_end_to_end import Model_end_to_end
from models.optimization.utils_optimization import get_solver_solution, smaller_crop
from analysis.visualize_results.plot_wavefield import plot_wavefield_optimization
from analysis.visualize_results.plot_heatmap import plot_heatmap_optimization
from generate_data.utils_wave import crop_center
import torch
import numpy as np
from models.model_utils import get_params
from models.optimization.parallel_scheme import parareal_scheme
from models.optimization.parallel_procrustes_scheme import parareal_procrustes_scheme


def vis_parareal(vel_name, big_vel, n_snaps, mode="parareal"):

    # data
    big_vel = torch.from_numpy(big_vel)
    res = big_vel.shape[-1]
    vel = crop_center(big_vel.squeeze(),128)
    u_0 = torch.concat([initial_condition_gaussian(big_vel,res=res, mode=mode),
                        big_vel.unsqueeze(dim=0)], dim=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_end_to_end(param_dict=get_params("0")).double().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10_2.pt'))
    model.eval()

    with torch.no_grad():
        parareal_tensor = choose_optimization_method(model, u_0,big_vel,n_snaps,mode=mode)  # k x s x b x c x w x h
        coarse_solver_tensor = get_solver_solution(smaller_crop(u_0[:, :3, :, :]), 11,smaller_crop(u_0[:, 3, :,:]).unsqueeze(dim=0), solver="coarse")  # s x b x c x w x h
        fine_solver_tensor = get_solver_solution(u_0[:, :3, :, :], 11, u_0[:, 3, :, :].unsqueeze(dim=0), solver="fine")  # s x b x c x w x h
        ticks = get_ticks_fine(fine_solver_tensor, vel)  # s x 3

        plot_wavefield_optimization(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, vel, vel_name, mode)
        plot_heatmap_optimization(fine_solver_tensor, parareal_tensor, vel, vel_name, mode)

        np.save('../../results/optimization/' + mode+ "/"+vel_name + '_coarse.npy', coarse_solver_tensor.numpy())
        np.save('../../results/optimization/' + mode+ "/"+vel_name + '_fine.npy', fine_solver_tensor.numpy())
        np.save('../../results/optimization/' + mode+ "/"+vel_name + '_parareal.npy', parareal_tensor.numpy())
        np.save('../../results/optimization/' + mode+ "/"+vel_name + '_velocity_profile.npy', big_vel.numpy())


def choose_optimization_method(model,u_0,big_vel,n_snaps,mode="parareal"):
    if mode == "parareal":
        return parareal_scheme(model, u_0, big_vel, n_snaps=n_snaps)
    elif mode == "parareal_procrustes":
        return parareal_procrustes_scheme(model, u_0, big_vel, n_snaps)
    else:
        raise NotImplementedError("This optimization method has not been implemented yet!")


def vis_multiple_init_cond():
    res = 256
    velocities = get_velocity_dict(res, 1, "../../data/crops_bp_m_200_256.npz")
    for vel_name, vel in velocities.items():
        print("-"*20,vel_name,"-"*20)
        vis_parareal(vel_name, vel, mode="parareal", n_snaps=7)


if __name__ == '__main__':
    vis_multiple_init_cond()