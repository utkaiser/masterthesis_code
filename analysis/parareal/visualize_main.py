import sys
sys.path.append("..")
from models.parallel_procrustes_scheme import one_iteration_velocity_verlet
import torch.nn.functional as F
from scipy.io import loadmat
from skimage.filters import gaussian
from analysis.parareal.plot_wavefield import plot_wavefield_results, get_ticks_fine
from analysis.parareal.plot_heatmap import plot_wavefield_heatmap
from generate_data.initial_conditions import init_gaussian_parareal, diagonal_ray, three_layers, crack_profile, \
    high_frequency
from generate_data.wave_util import crop_center
import torch
import numpy as np
from models import model_end_to_end
from models.model_utils import get_params
from models.parallel_scheme import smaller_crop, one_iteration_pseudo_spectral, parareal_scheme
from models.parallel_procrustes_scheme import parareal_procrustes_scheme
#from models.parallel_scheme import parareal_scheme


def vis_parareal(vel_name, big_vel, folder_name):

    # data
    vel = crop_center(big_vel,128,128)
    u_0 = torch.concat([init_gaussian_parareal(256,big_vel), big_vel.unsqueeze(dim=0).unsqueeze(dim=0)],dim=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_end_to_end.Restriction_nn(param_dict=get_params("0")).double().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10_2.pt'))
    model.eval()

    MSE_loss = torch.nn.MSELoss()

    with torch.no_grad():
        parareal_tensor = parareal_procrustes_scheme(model, u_0,big_vel)  # k x s x b x c x w x h  # TODO: decide if parareal or parareal + procrustes
        coarse_solver_tensor = get_solver_solution(smaller_crop(u_0[:, :3, :, :]), 11,smaller_crop(u_0[:, 3, :,:]).unsqueeze(dim=0), solver="coarse")  # s x b x c x w x h
        fine_solver_tensor = get_solver_solution(u_0[:, :3, :, :], 11,u_0[:, 3, :, :].unsqueeze(dim=0), solver="fine")  # s x b x c x w x h
        ticks = get_ticks_fine(fine_solver_tensor, vel)  # s x 3

        plot_wavefield_results(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, MSE_loss, vel, vel_name, folder_name)
        plot_wavefield_heatmap(coarse_solver_tensor, fine_solver_tensor, parareal_tensor, ticks, MSE_loss, vel, vel_name, folder_name)

        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_coarse.npy', coarse_solver_tensor.numpy())
        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_fine.npy', fine_solver_tensor.numpy())
        np.save('../../results/parareal/' + folder_name+ "/"+vel_name + '_parareal.npy', parareal_tensor.numpy())


def get_solver_solution(u_n_k, n_snapshots, vel, solver="coarse"):
    # u_0_k -> b x c x w x h
    # vel -> b x w x h

    if solver == "coarse":
        small_res_scale = 2
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w, h])

        for s in range(n_snapshots):

            sol[s] = u_n_k

            a = F.upsample(u_n_k[:,0].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b = F.upsample(u_n_k[:,1].unsqueeze(dim=0), size=(w//small_res_scale, w//small_res_scale), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w // small_res_scale, w // small_res_scale), mode='bilinear')
            d = F.upsample(vel, size=(w//small_res_scale, w//small_res_scale), mode='bilinear')

            u_n_k = torch.concat([a,b,b2,d],dim=1)

            u_n_k = one_iteration_velocity_verlet(u_n_k,f_delta_x=2./64., f_delta_t=1./600., delta_t_star = .06)

            a2 = F.upsample(u_n_k[:, 0].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b2 = F.upsample(u_n_k[:, 1].unsqueeze(dim=0), size=(w, w), mode='bilinear')
            b22 = F.upsample(u_n_k[:, 2].unsqueeze(dim=0), size=(w, w), mode='bilinear')

            u_n_k = torch.concat([a2,b2,b22], dim=1)


    elif solver == "fine":
        b, c, w, h = u_n_k.shape
        sol = torch.zeros([n_snapshots, b, c, w//2, h//2])

        for s in range(n_snapshots):
            sol[s] = smaller_crop(u_n_k)
            u_n_k = torch.concat([u_n_k,vel], dim=1)
            u_n_k = one_iteration_pseudo_spectral(u_n_k)

    else:
        raise NotImplementedError("This solver has not been implemented yet.")

    return sol


def get_velocity_crop(resolution, velocity_profile):

    if velocity_profile == "diagonal":
        img = diagonal_ray(1,res=resolution).squeeze()

    elif velocity_profile == "marmousi":
        datamat = loadmat('../../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        img = gaussian(datamat['marm1larg'], 4)  # to make smoother
        img = img[200:200+resolution,200:200+resolution]

    elif velocity_profile == "marmousi2":
        datamat = loadmat('../../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        img = gaussian(datamat['marm1larg'], 4)  # to make smoother
        img = img[300:300+resolution,300:300+resolution]

    elif velocity_profile == "bp":
        databp = loadmat('../../data/bp2004.mat')  # velocity models BP dataset
        img = gaussian(databp['V'], 4) / 1000  # to make smoother (and different order of magnitude)
        img = img[1100:1100 + resolution, 1100:1100 + resolution]

    elif velocity_profile == "three_layers":
        img = three_layers(1, res=resolution).squeeze()

    elif velocity_profile == "crack_profile":
        img = crack_profile(1, res=resolution).squeeze()

    elif velocity_profile == "high_frequency":
        img = high_frequency(1, res=resolution).squeeze()

    else:
        raise NotImplementedError("Velocity model not implemented.")

    return img



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
    for key, vel in velocities.items():
        print(key,"-"*20)
        vis_parareal(key, vel, folder_name = "procrustes")


if __name__ == '__main__':
    vis_multiple_init_cond()