import torch
from models.model_utils import flip_tensors, sample_label_normal_dist
from models.optimization.utils_optimization import one_iteration_pseudo_spectral, smaller_crop, \
    one_iteration_pseudo_spectral_tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from analysis.visualize_results.visualize_training import visualize_wavefield
import random
import datetime


def model_optimization_solution(data, model, loss_f, n_snaps, label_distr_shift, mode, i, epoch,
                                vis_path, vis_save, optimization_type, multi_step):

    loss_list = []

    if mode == "train":

        for input_idx in random.choices(range(n_snaps - 2), k=n_snaps):
            label_range = sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, multi_step)

            if optimization_type is None:
                input_tensor = data[:, input_idx]  # b x 4 x w x h
                for label_idx in range(input_idx + 1, label_range):  # randomly decide how long path is
                    label = data[:, label_idx, :3].to(device)  # b x 3 x w x h
                    output = model(input_tensor)  # b x 3 x w x h
                    loss_list += [loss_f(output, label)]
                    input_tensor = torch.cat((output, input_tensor[:, 3].unsqueeze(dim=1)), dim=1)

            elif optimization_type == "parareal":
                print(datetime.datetime.now(), input_idx,label_range)
                loss_list += parareal_scheme(model, input_idx, 2, label_range, loss_f, data.to(device).detach())

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


def parareal_scheme(model, input_idx, n_parareal, label_range, loss_f, data):
    # data -> b x n_snaps x 4 x w x h

    # data setup
    u_0 = data[:, input_idx]  # b x 4 x w x h
    u_n = u_0.clone()  # b x 4 x w x h
    vel = u_0[:,3].unsqueeze(dim=1).clone()  # b x 1 x 256 x 256
    batch_size, channel, width, height = u_n.shape  # b x 4 x 256 x 256
    big_tensor = torch.zeros([batch_size, label_range - input_idx + 1, channel - 1, width, height])
    loss_list = []

    # initial guess, first iteration without parareal
    for n in range(label_range - input_idx):
        u_n1 = model(u_n)  # b x c x w x h
        loss_list.append(loss_f(smaller_crop(u_n1),
                                smaller_crop(data[:, n+1, :3].to(device))))
        big_tensor[:, n+1] = u_n1.clone()
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        big_tensor[:, 0] = u_0[:, :3].clone()
        res_fine, res_model = get_optimizing_terms(model, big_tensor, vel, input_idx, label_range)  # n_snaps x b x c x w x h
        new_big_tensor = torch.zeros([batch_size, label_range - input_idx + 1, channel - 1, width, height])
        new_big_tensor[:, 0] = u_0[:, :3].clone()

        for n in range(label_range - input_idx):
            u_n_k1 = torch.cat((new_big_tensor[:, n], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + (res_fine[:,n] - res_model[:,n])
            loss_list.append(loss_f(smaller_crop(u_n1_k1),
                                    smaller_crop(data[:, input_idx + n + 1, :3].to(device))))
            new_big_tensor[:, n+1] = u_n1_k1

        big_tensor = new_big_tensor.detach().clone()

    return loss_list


def get_optimizing_terms(model, big_pseudo_tensor, vel, input_idx, label_range):
    # this can be done later computed in parallel
    # big_pseudo_tensor -> b x diff x c x w x h
    # vel -> b x 1 x w x h

    model.eval()
    with torch.no_grad():
        bs, n_snaps, c, w, h = big_pseudo_tensor.shape
        res_fine = torch.zeros([bs, n_snaps, c, w, h]).double()
        res_model = torch.zeros([bs, n_snaps, c, w, h]).double()
        for s in range(label_range - input_idx):
            res_fine[:,s], res_model[:,s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[:,s], vel], dim=1))

    model.train()
    return res_fine, res_model


def compute_parareal_term(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # b x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral_tensor(u_n_k)  # b x 3 x w x h
    return res_fine_solver, res_model






