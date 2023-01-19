import torch
from generate_data.wave_propagation import pseudo_spectral
from generate_data.wave_util import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor


def parareal_scheme(model, u_0, _, n_parareal = 4, n_snapshots = 11):

    # u_0 -> b x c x w x h

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
        res_fine, res_model = get_optimizing_terms(model, big_tensor, n_snapshots, vel)  # n_snapshots x b x c x w x h
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
        new_big_tensor[0] = u_0[:, :3].clone()

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((new_big_tensor[n], vel), dim=1)
            u_n1_k1 = model(u_n_k1).squeeze() + (res_fine[n] - res_model[n])
            parareal_tensor[k, n+1] = smaller_crop(u_n1_k1)
            new_big_tensor[n+1] = u_n1_k1

        big_tensor = new_big_tensor.clone()

    return parareal_tensor  # k x s x b x c x w x h


def get_optimizing_terms(model, big_pseudo_tensor, _, vel):
    # this can be done later computed in parallel

    n_snapshots, b, c, w, h = big_pseudo_tensor.shape
    res_fine = torch.zeros([n_snapshots, c, w, h]).double()
    res_model = torch.zeros([n_snapshots, c, w, h]).double()  # n_snapshots x c x w x h

    for s in range(n_snapshots):
        res_fine[s], res_model[s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[s], vel], dim=1))

    return res_fine, res_model


def compute_parareal_term(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # 1 x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral(u_n_k)  # 1 x 3 x w x h

    return res_fine_solver.squeeze(), res_model.squeeze()


def smaller_crop(matrix):
    # matrix -> b? x c x w x h
    v = 64
    if len(matrix.shape) == 4:
        return matrix[:,:,v:-v, v:-v]
    elif len(matrix.shape) == 3:
        return matrix[:, v:-v, v:-v]
    else:
        raise NotImplementedError("This dimensionality has not been implemented yet.")


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





