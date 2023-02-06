from analysis.visualize_results.visualize_training import visualize_wavefield
import random
import torch
from models.model_utils import sample_label_normal_dist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.optimization_models.parallel_scheme import parareal_scheme


def model_optimization_solution(data, model, loss_f, n_snaps, label_distr_shift, mode, i, epoch,
                                vis_path, vis_save, optimization_type, multi_step):

    loss_list = []

    if mode == "train":
        for input_idx in random.choices(range(n_snaps - 2), k=n_snaps):
            label_range = sample_label_normal_dist(input_idx, n_snaps, label_distr_shift, multi_step)

            if optimization_type == "parareal":
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