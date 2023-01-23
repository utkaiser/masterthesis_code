import torch
from generate_data import opp_model
from generate_data.wave_propagation import velocity_verlet_tensor
from generate_data.wave_util import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
from models.parallel_scheme import one_iteration_pseudo_spectral, smaller_crop
from generate_data.postprocess_wave import ApplyOPP2WaveSol
import matplotlib.pyplot as plt

def parareal_procrustes_scheme(model, u_0, _, n_parareal = 4, n_snapshots = 10):
    # u_0 -> b x c x w x h

    u_n = u_0.clone()
    p_u_n = u_0.clone()
    vel = u_n[:,3].clone().unsqueeze(dim=1)  # 1 x 1 x 256 x 256
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 256 x 256
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, 128, 128])
    big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
    p_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

    # initial guess, first iteration without parareal
    parareal_tensor[0, 0] = smaller_crop(u_n[:, :3].clone())
    big_tensor[0] = u_0[:, :3].clone()
    p_tensor[0] = u_0[:, :3].clone()
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 512 x 512
        parareal_tensor[0,n+1] = smaller_crop(u_n1)
        big_tensor[n + 1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

        p_u_n1 = one_iteration_pseudo_spectral(p_u_n)
        p_tensor[n + 1] = p_u_n1
        p_u_n = torch.cat([p_u_n1, vel], dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print(k)

        res_fine, res_model = get_optimizing_terms(model, big_tensor, n_snapshots, vel)  # n_snapshots x b x c x w x h

        if k == 1:
            P, S, Q = opp_model.ProcrustesShiftMap(coarse_dat=(big_tensor[:,0,0].permute(1, 2, 0), big_tensor[:,0,1].permute(1, 2, 0), big_tensor[:,0,2].permute(1, 2, 0)),
                                                   fine_dat=(p_tensor[:,0,0].permute(1, 2, 0), p_tensor[:,0,1].permute(1, 2, 0), p_tensor[:,0,2].permute(1, 2, 0)))
        else:
            P, S, Q = opp_model.ProcrustesShiftMap(coarse_dat=(big_tensor[:,0,0].permute(1, 2, 0), big_tensor[:,0,1].permute(1, 2, 0), big_tensor[:,0,2].permute(1, 2, 0)),
                                                   fine_dat=(p_tensor[:,0,0].permute(1, 2, 0), p_tensor[:,0,1].permute(1, 2, 0), p_tensor[:,0,2].permute(1, 2, 0)),
                                                   opmap=(P, S, Q))

        parareal_tensor[k, 0] = smaller_crop(u_0[:, :3].clone())
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
        new_big_tensor[0] = u_0[:, :3].clone()

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((new_big_tensor[n], vel), dim=1)
            u_n1_k1 = parareal_procrust(model, res_fine[n], big_tensor[n], n_snapshots, vel, u_n_k1, P,S,Q)
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


def parareal_procrust(model, res_fine, big_tensor, _, vel, u_n_k1, P,S,Q, dx = 2.0 / 128.0):
    # this can be done later computed in parallel

    # pseudo-spectral
    u_n1_k_pseudo = model(u_n_k1)
    u_n1_k_pseudo_u, u_n1_k_pseudo_ut = WaveSol_from_EnergyComponent_tensor(u_n1_k_pseudo[:, 0, :, :],
                                             u_n1_k_pseudo[:, 1, :, :],
                                             u_n1_k_pseudo[:, 2, :, :],
                                             vel,dx,torch.sum(torch.sum(torch.sum(u_n1_k_pseudo[:, 0, :, :]))))
    u_n1_k_pseudo_u, u_n1_k_pseudo_ut = ApplyOPP2WaveSol(u_n1_k_pseudo_u.squeeze().numpy(),
                                                        u_n1_k_pseudo_ut.squeeze().numpy(),
                                                        vel.squeeze().numpy(), dx, (P, S, Q))
    u_n1_k_pseudo_u_x, u_n1_k_pseudo_u_y, u_n1_k_pseudo_u_t_c = WaveEnergyComponentField_tensor(torch.from_numpy(u_n1_k_pseudo_u).unsqueeze(dim=0),
                                                                                                torch.from_numpy(u_n1_k_pseudo_ut).unsqueeze(dim=0), vel,dx)
    res_pseudo = torch.concat([u_n1_k_pseudo_u_x, u_n1_k_pseudo_u_y, u_n1_k_pseudo_u_t_c], dim=0)

    # model
    u_n1_k_model = model(torch.concat([big_tensor,vel], dim=1))
    u_n1_k_model_u, u_n1_k_model_ut = WaveSol_from_EnergyComponent_tensor(u_n1_k_model[:, 0, :, :],
                                                                            u_n1_k_model[:, 1, :, :],
                                                                            u_n1_k_model[:, 2, :, :],
                                                                            vel, dx,
                                                                            torch.sum(torch.sum(torch.sum(u_n1_k_model[:, 0, :, :]))))
    u_n1_k_model_u, u_n1_k_model_ut = ApplyOPP2WaveSol(u_n1_k_model_u.squeeze().numpy(),
                                                        u_n1_k_model_ut.squeeze().numpy(),
                                                        vel.squeeze().numpy(), dx, (P, S, Q))
    u_n1_k_model_u_x, u_n1_k_model_u_y, u_n1_k_model_u_t_c = WaveEnergyComponentField_tensor(torch.from_numpy(u_n1_k_model_u).unsqueeze(dim=0),
                                                                                                torch.from_numpy(u_n1_k_model_ut).unsqueeze(dim=0), vel,dx)
    res_model = torch.concat([u_n1_k_model_u_x, u_n1_k_model_u_y, u_n1_k_model_u_t_c], dim=0)

    return res_pseudo + (res_fine - res_model)


def one_iteration_velocity_verlet(u_n_k,f_delta_x=2./128., f_delta_t=(2./128.)/20.,delta_t_star=.06):
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





