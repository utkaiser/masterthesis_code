import torch
from analysis.visualize_results.plot_training_optimization import plot_big_tensor
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor
from models.model_utils import smaller_crop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parareal_scheme(model, input_idx, n_parareal, label_range, loss_f, data):
    # data -> b x n_snaps x 4 x w x h

    # data setup
    u_0 = data[:, input_idx]  # b x 4 x w x h
    u_n = u_0.clone()  # b x 4 x w x h
    vel = u_0[:,3].unsqueeze(dim=1).clone()  # b x 1 x 256 x 256
    batch_size, channel, width, height = u_n.shape  # b x 4 x 256 x 256
    big_tensor = torch.zeros([n_parareal+1, batch_size, label_range - input_idx + 1, channel - 1, width, height])
    loss_list = []

    # initial guess, first iteration without parareal
    big_tensor[0, :, 0] = u_0[:,:3].clone()
    for n in range(label_range - input_idx):
        u_n1 = model(u_n)  # b x c x w x h
        loss_list.append(loss_f(smaller_crop(u_n1),
                                smaller_crop(data[:, n+1, :3].to(device))))
        big_tensor[0, :, n+1] = u_n1.detach().clone()
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        big_tensor[k, :, 0] = u_0[:, :3].clone()
        res_fine, res_model = get_optimizing_terms(model, big_tensor[k-1], vel, input_idx, label_range)  # bs x n_snaps x c x w x h
        new_big_tensor = torch.zeros([batch_size, label_range - input_idx + 1, channel - 1, width, height])
        new_big_tensor[:, 0] = u_0[:, :3].clone()

        for n in range(label_range - input_idx):
            u_n_k1 = torch.cat((new_big_tensor[:, n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + (res_fine[:,n] - res_model[:,n])

            # we train only when parareal it <= curr snapshot; since for all other cases, pseudo-spectral is applied n times, so close perfect result already
            if k <= n:
                loss_list.append(loss_f(smaller_crop(u_n1_k1),
                                        smaller_crop(data[:, input_idx + n + 1, :3].to(device))))
            new_big_tensor[:, n+1] = u_n1_k1

        big_tensor[k] = new_big_tensor.detach().clone()
    plot_big_tensor(smaller_crop(big_tensor), smaller_crop(vel), smaller_crop(data[:,input_idx:label_range+1]))

    return loss_list


def get_optimizing_terms(model, big_pseudo_tensor, vel, input_idx, label_range):
    # this can be done later computed in parallel
    # big_pseudo_tensor -> b x diff x c x w x h
    # vel -> b x 1 x w x h

    bs, n_snaps, c, w, h = big_pseudo_tensor.shape
    res_fine = torch.zeros([bs, n_snaps, c, w, h]).double()
    res_model = torch.zeros([bs, n_snaps, c, w, h]).double()

    model.eval()
    with torch.no_grad():
        for s in range(label_range - input_idx):
            res_fine[:,s], res_model[:,s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[:,s], vel], dim=1), vel)

    model.train()
    return res_fine, res_model


def compute_parareal_term(model, u_n_k, vel):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # b x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral_tensor(u_n_k, f_delta_x=2./128., f_delta_t=(2./128.) / 40.)  # b x 3 x w x h

    return res_fine_solver, res_model


def one_iteration_pseudo_spectral_tensor(u_n_k, f_delta_x = 2./128., f_delta_t = (2./128.) / 20., delta_t_star = .06):
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



