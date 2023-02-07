import torch
from matplotlib import pyplot as plt
from analysis.utils_analysis import one_iteration_pseudo_spectral
from analysis.visualize_results.plot_training_optimization import plot_big_tensor
from generate_data.utils_wave import WaveSol_from_EnergyComponent_tensor, WaveEnergyComponentField_tensor
from generate_data.wave_propagation import pseudo_spectral_tensor
from models.model_utils import smaller_crop, get_wavefield, compute_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parareal_scheme(model, u_0, fine_sol, n_parareal = 2, n_snapshots = 7):

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
            new_big_tensor[0] = u_0[:, :3].clone()

            for n in range(n_snapshots-1):
                u_n_k1 = torch.cat((new_big_tensor[n], vel), dim=1)
                u_n1_k1 = model(u_n_k1) + parareal_terms[n]
                parareal_tensor[k, n+1] = smaller_crop(u_n1_k1)
                new_big_tensor[n+1] = u_n1_k1

            big_tensor = new_big_tensor.clone()
        a(parareal_tensor, vel, fine_sol)
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


def parareal_scheme2(model, input_idx, n_parareal, label_range, loss_f, fine_tensor, vel):
    # data -> b x n_snaps x 3 x w x h
    n_parareal = 2
    # data setup
    u_0 = fine_tensor[:, input_idx]  # b x 3 x w x h
    vel = vel.unsqueeze(dim=0).unsqueeze(dim=0)
    u_n = torch.cat((u_0.clone(), vel), dim=1)  # b x 4 x w x h
    batch_size, channel, width, height = u_0.shape  # b x 4 x 256 x 256
    big_tensor = torch.zeros([n_parareal+1, batch_size, label_range - input_idx + 1, channel, width, height]).to(device)
    loss_list = []

    # initial guess, first iteration without parareal
    big_tensor[0, :, 0] = u_0[:,:3].clone()

    for n in range(label_range - input_idx):
        u_n1 = model(u_n)  # b x c x w x h
        loss_list.append(loss_f(smaller_crop(u_n1),
                                smaller_crop(fine_tensor[:, n + 1,:3])))

        big_tensor[0, :, n+1] = u_n1.clone().detach()
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print("--- parareal it",k)
        res_fine, res_model = get_optimizing_terms(model, big_tensor[k-1].to(device), vel.to(device), input_idx, label_range)  # bs x n_snaps x c x w x h
        new_big_tensor = torch.zeros([batch_size, label_range - input_idx + 1, channel, width, height]).to(device)
        new_big_tensor[:, 0] = u_0[:, :3].clone()

        for n in range(label_range - input_idx):
            u_n_k1 = torch.cat((new_big_tensor[:, n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + res_fine[:,n] - res_model[:,n]

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
        res_fine[:,s], res_model[:,s] = compute_parareal_term2(model, torch.cat([big_pseudo_tensor[:,s], vel], dim=1))

    #model.train()
    return res_fine, res_model


def compute_parareal_term2(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # b x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral_tensor(u_n_k)  # b x 3 x w x h

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



def a(parareal_tensor, vel, fine_sol):

    fig = plt.figure(figsize=(35, 15))

    for k in range(parareal_tensor.shape[0]):
        for s in range(parareal_tensor.shape[1]):
            wave_field = get_wavefield(parareal_tensor[k,s], smaller_crop(vel))
            ax = fig.add_subplot(3, 11, 11 * k + s + 1)
            pos = ax.imshow(wave_field)
            if s != 0:
                ax.set_title(compute_loss(parareal_tensor[k,s], smaller_crop(fine_sol[0,s].unsqueeze(dim=0)), smaller_crop(vel)))
            plt.axis('off')

    plt.show()