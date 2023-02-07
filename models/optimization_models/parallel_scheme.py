import torch
from matplotlib import pyplot as plt
from analysis.visualize_results.plot_training_optimization import plot_big_tensor
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor
from models.model_utils import smaller_crop, get_wavefield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parareal_scheme(model, input_idx, n_parareal, label_range, loss_f, fine_tensor):
    # data -> b x n_snaps x 3 x w x h
    n_parareal = 2
    f_delta_x = 2./128.
    # data setup
    u_0 = fine_tensor[:, input_idx]  # b x 3 x w x h
    vel = fine_tensor[:, 0, 2].unsqueeze(dim=0)
    u_n = u_0.clone()  # b x 3 x w x h
    batch_size, channel, width, height = u_0.shape  # b x 4 x 256 x 256
    big_tensor = torch.zeros([n_parareal+1, batch_size, label_range - input_idx + 1, 2, width, height]).to(device)
    loss_list = []

    # initial guess, first iteration without parareal
    big_tensor[0, :, 0] = u_0[:,:2].clone()

    for n in range(label_range - input_idx):
        u_n = up3(u_n, vel)
        u_n1 = model(u_n)  # b x c x w x h
        loss_list.append(loss_f(smaller_crop(u_n1),
                                smaller_crop(up3(fine_tensor[:, n + 1,:2], vel))))

        big_tensor[0, :, n+1] = u_n1.clone().detach()
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print("--- parareal it",k)
        res_fine, res_model = get_optimizing_terms(model, big_tensor[k-1].to(device), vel.to(device), input_idx, label_range)  # bs x n_snaps x c x w x h
        new_big_tensor = torch.zeros([batch_size, label_range - input_idx + 1, 2, width, height]).to(device)
        new_big_tensor[:, 0] = u_0[:, :2].clone()

        for n in range(label_range - input_idx):
            u_n_k1 = torch.cat((new_big_tensor[:, n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + res_fine[:,n] - res_model[:,n]

            fig = plt.figure(figsize=(30, 15))
            ax = fig.add_subplot(1,3,1)
            ax.imshow(get_wavefield(model(u_n_k1).clone().detach(), vel.squeeze()))
            ax1 = fig.add_subplot(1, 3, 2)
            ax1.imshow(get_wavefield(res_fine[0,n].unsqueeze(dim=0).clone().detach(), vel.squeeze()))
            ax2 = fig.add_subplot(1, 3, 3)
            ax2.imshow(get_wavefield(res_model[0,n].unsqueeze(dim=0).clone().detach(), vel.squeeze()))
            plt.show()

            # we train only when parareal it <= curr snapshot; since for all other cases, pseudo-spectral is applied n times, so close perfect result already
            # if k <= n: loss_list.append(loss_f(smaller_crop(u_n1_k1),
            #                             smaller_crop(fine_tensor[:, n + 1].to(device))))

            new_big_tensor[:, n+1] = u_n1_k1

        big_tensor[k] = new_big_tensor.clone().detach()
    plot_big_tensor(smaller_crop(big_tensor), smaller_crop(vel), smaller_crop(fine_tensor))

    return loss_list


def get_optimizing_terms(model, big_pseudo_tensor, vel, input_idx, label_range):
    # this can be done later computed in parallel
    # big_pseudo_tensor -> b x diff x c x w x h
    # vel -> b x 1 x w x h

    bs, n_snaps, c, w, h = big_pseudo_tensor.shape
    res_fine = torch.zeros([bs, n_snaps, c, w, h]).double()
    res_model = torch.zeros([bs, n_snaps, c, w, h]).double()

    #model.eval()
    #with torch.no_grad():
    for s in range(label_range - input_idx):
        res_fine[:,s], res_model[:,s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[:,s], vel], dim=1))

    #model.train()
    return res_fine, res_model


def compute_parareal_term(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # b x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral_tensor(u_n_k)  # b x 3 x w x h

    return res_fine_solver, res_model


def one_iteration_pseudo_spectral_tensor(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 50., delta_t_star = .06):
    # u_n_k -> b x c x w x h
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n_k[:, 0],
                                                 u_n_k[:, 1],
                                                 u_n_k[:, 2],
                                                 u_n_k[:, 3],
                                                 f_delta_x,
                                                 torch.sum(torch.sum(torch.sum(u_n_k[:, 0]))))
    vel = u_n_k[:, 3]
    u_prop, u_t_prop = pseudo_spectral_tensor(u, u_t, vel, f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                      u_t_prop,
                                                      vel, f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def one_iteration_pseudo_spectral_tensor_ut(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 50., delta_t_star = .06):
    # u_n_k -> b x c x w x h

    u_prop, u_t_prop = pseudo_spectral_tensor(u_n_k[:,0], u_n_k[:,1], u_n_k[:, 2], f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(u_prop,
                                                      u_t_prop,
                                                      u_n_k[:, 2], f_delta_x)
    return torch.stack([u_x, u_y, u_t_c], dim=1)


def up3(u_n, vel, f_delta_x = 2./128.):
    u, ut = u_n[:,0], u_n[:,1]
    a,b,c = WaveEnergyComponentField_tensor(u, ut, vel, f_delta_x)
    return torch.stack([a,b,c], dim=1)




