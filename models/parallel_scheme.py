import scipy
import torch
from scipy.io import loadmat
from skimage.filters import gaussian
from generate_data.initial_conditions import init_cond_gaussian, init_gaussian_parareal, diagonal_ray
from models import model_end_to_end
from generate_data.wave_propagation import pseudo_spectral, velocity_verlet_tensor
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor, \
    WaveEnergyField_tensor, crop_center
import matplotlib.pyplot as plt
from models.model_utils import fetch_data_end_to_end, get_params

def parareal_scheme(model, u_0, big_vel, n_parareal = 4, n_snapshots = 11):

    # u_n -> b x c x w x h

    u_n = u_0.clone()
    vel = u_n[:,3].clone().unsqueeze(dim=1)  # 1 x 1 x 500 x 500
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 500 x 500
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, 128, 128])
    big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

    # initial guess, first iteration without parareal
    parareal_tensor[0, 0] = smaller_crop(u_n[:, :3].clone())
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 512 x 512
        parareal_tensor[0,n+1] = smaller_crop(u_n1)
        big_tensor[n+1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print(k)

        big_tensor[0] = u_0[:, :3].clone()
        parareal_tensor[k, 0] = smaller_crop(u_0[:, :3].clone())
        parareal_terms = get_parareal_terms(model, big_tensor, n_snapshots, vel) # n_snapshots x b x c x w x h
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((parareal_tensor[k,n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + parareal_terms[n]
            parareal_tensor[k, n+1] = smaller_crop(u_n1_k1)
            new_big_tensor[n+1] = u_n1_k1

        big_tensor = new_big_tensor.clone()

    return parareal_tensor  # k x s x b x c x w x h


def get_parareal_terms(model, big_pseudo_tensor, n_snapshots, vel):
    # this can be done later computed in parallel

    parareal_terms = torch.zeros(big_pseudo_tensor.shape) # n_snapshots x b x c x w x h

    for s in range(n_snapshots):
        parareal_terms[s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[s], vel], dim=1))

    return parareal_terms


def compute_parareal_term(model, u_n_k):

    # u_n_k -> b x c x w x h

    res_fine_solver = one_iteration_pseudo_spectral(u_n_k) #one_iteration_velocity_verlet(u_n_k)
    res_model = model(u_n_k)  # procrustes_optimization(model(u_n_k), res_fine_solver)

    return res_fine_solver - res_model

def one_iteration_pseudo_spectral(u_n_k):

    # u_n_k -> b x c x w x h

    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    delta_t_star = .06

    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0, :, :].clone(),
                                                 u_n_k[:, 1, :, :].clone(),
                                                 u_n_k[:, 2, :, :].clone(),
                                                 u_n_k[:, 3, :, :].clone(),
                                                 f_delta_x,
                                                 torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))))
    u, u_t, vel = u.squeeze().numpy(), u_t.squeeze().numpy(), u_n_k[:, 3, :, :].clone().squeeze().numpy()
    u_prop, u_t_prop = pseudo_spectral(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(torch.from_numpy(u_prop).unsqueeze(dim=0),
                                                      torch.from_numpy(u_t_prop).unsqueeze(dim=0),
                                                      torch.from_numpy(vel).unsqueeze(dim=0), f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def procrustes_optimization(matrix, target):
    # matrix -> n_snapshots x 1 x 4 x 128 x 128

    procrustes_res = torch.zeros(matrix.shape)

    # channel-wise procrustes
    for c in range(matrix.shape[1]):
        m, t = matrix[0,c], target[0,c]
        omega, _ = scipy.linalg.orthogonal_procrustes(m, t)
        procrustes_res[0,c,:,:] = torch.from_numpy(omega) * m

    return procrustes_res


def one_iteration_velocity_verlet(u_n_k, f_delta_x = 2.0 / 128.0, f_delta_t = (2.0 / 128.0) / 20, delta_t_star = .06):

    # u_n_k -> b x c x w x h

    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0, :, :].clone(),
                                                 u_n_k[:, 1, :, :].clone(),
                                                 u_n_k[:, 2, :, :].clone(),
                                                 u_n_k[:, 3, :, :].clone(),
                                                 f_delta_x,
                                                 torch.sum(torch.sum(torch.sum(u_n_k[:, 0, :, :].clone()))))
    vel = u_n_k[:, 3, :, :].clone()
    u_prop, u_t_prop = velocity_verlet_tensor(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star,number=1,boundary_c="absorbing")
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                      u_t_prop,
                                                      vel, f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def get_velocity_crop(resolution, diagonal=False):

    if diagonal:
        img = diagonal_ray(1,res=resolution).squeeze()

    else:
        datamat = loadmat('../data/marm1nonsmooth.mat')  # velocity models Marmousi dataset
        img = gaussian(datamat['marm1larg'], 4)  # to make smoother
        img = img[200:200+resolution,200:200+resolution]

    return img


def round_loss(number):
    return number #str(round(number*(10**7),5))+"e-7"

def smaller_crop(matrix):
    # matrix -> b x c x w x h
    return matrix  # [:,:,v:-v, v:-v]


def get_wavefield(tensor, vel, f_delta_x = 2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                           f_delta_x) * f_delta_x * f_delta_x

import torch.nn.functional as F

def get_solver_solution(u_n_k, n_snapshots, vel, solver="coarse"):
    # u_0_k -> b x c x w x h
    # vel -> b x w x h

    b, c, w, h = u_n_k.shape
    sol = torch.zeros([n_snapshots, b, c, w, h])

    if solver == "coarse":
        small_res_scale = 2

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
        for s in range(n_snapshots):
            sol[s] = u_n_k
            u_n_k = torch.concat([u_n_k,vel], dim=1)
            u_n_k = one_iteration_pseudo_spectral(u_n_k)

    return sol


def vis_parareal():

    big_vel = torch.from_numpy(get_velocity_crop(128, diagonal=True))
    vel = crop_center(big_vel,128,128)
    u_0 = torch.concat([init_gaussian_parareal(128,big_vel), big_vel.unsqueeze(dim=0).unsqueeze(dim=0)],dim=1)

    # data
    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    param_dict = get_params("0")
    # path = ['../data/end_to_end_bp_m_10_2000.npz']
    # loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=True, batch_size=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_end_to_end.Restriction_nn(param_dict=param_dict).double().to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10_2.pt'))
    model.eval()
    MSE_loss = torch.nn.MSELoss()

    fig = plt.figure(figsize=(35, 15))

    with torch.no_grad():
        coarse_solver_tensor = get_solver_solution(u_0[:, :3, :, :], 11,u_0[:, 3, :,:].unsqueeze(dim=0), solver="coarse")  # s x b x c x w x h
        fine_solver_tensor = get_solver_solution(u_0[:, :3, :, :], 11,u_0[:, 3, :, :].unsqueeze(dim=0), solver="fine")  # s x b x c x w x h
        parareal_tensor = parareal_scheme(model, u_0, big_vel)  # k x s x b x c x w_big x h_big

        # coarse solver solution
        for s in range(parareal_tensor.shape[1]):
            ax = fig.add_subplot(7, 11, 1 + s)
            wave_field = get_wavefield(coarse_solver_tensor[s], vel)
            pos = ax.imshow(wave_field)
            if s!=0:
                plt.colorbar(pos)
                ax.set_title(round_loss(MSE_loss(get_wavefield(fine_solver_tensor[s,:3], vel), wave_field).item()), fontdict={'fontsize': 9})
            plt.axis('off')

        # parareal scheme
        for k in range(parareal_tensor.shape[0]):
            for s in range(parareal_tensor.shape[1]):
                wave_field = get_wavefield(parareal_tensor[k,s], vel)
                ax = fig.add_subplot(7, 11, 11 + 11*k + s + 1)
                pos = ax.imshow(wave_field)
                if s != 0:
                    plt.colorbar(pos)
                    ax.set_title(round_loss(MSE_loss(get_wavefield(fine_solver_tensor[s,:3], vel), wave_field).item()), fontdict={'fontsize': 9})
                plt.axis('off')

        # fine solver solution
        for s in range(parareal_tensor.shape[1]):
            ax = fig.add_subplot(7, 11, 67 + s)
            wave_field = get_wavefield(fine_solver_tensor[s], vel)
            pos = ax.imshow(wave_field)
            if s != 0: plt.colorbar(pos)
            ax.set_title("fine solver it " + str(s),fontdict={'fontsize': 9})
            plt.axis('off')

        fig.suptitle("coarse solver (row 0), parareal end to end (k=0,...4) (row 1-5), fine solver (last row); titles represent MSE between result and fine solver")
        plt.show()




if __name__ == '__main__':
    vis_parareal()





