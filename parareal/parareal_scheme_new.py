import sys
import random

import numpy as np
import torch

from models.utils import sample_label_normal_dist

sys.path.append("..")
sys.path.append("../..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# todo: write this different and connect with already existing parareal code


def train_model(
    model,
    epoch,
    label_distr_shift,
    train_loader,
    param_d,
    global_step,
    logging_bool,
    optimizer,
    train_logger,
    loss_f,
    multi_step,
    weighted_loss,
):
    model.train()
    train_loss_list = []

    for i, data in enumerate(train_loader):
        loss_list = []
        data = data[0].to(device)  # b x n_snaps x 4 x w x h

        for input_idx in random.choices(
            range(param_d["n_snaps"]), k=param_d["n_snaps"]
        ):
            label_range = sample_label_normal_dist(
                input_idx,param_d["n_snaps"],0,False,False,
            )
            input_tensor = data[:, input_idx].detach()  # b x 4 x w x h

            for label_idx in range(
                input_idx + 1, label_range + 1
            ):  # randomly decide how long path is
                label = data[:, label_idx, :3]  # b x 3 x w x h
                output = model(input_tensor)  # b x 3 x w x h
                loss_list.append(loss_f(output, label))
                input_tensor = torch.cat(
                    (output, input_tensor[:, 3].unsqueeze(dim=1)), dim=1
                )

        optimizer.zero_grad()
        sum(loss_list).backward()
        optimizer.step()

        if logging_bool:
            train_logger.add_scalar(
                "loss", np.array(loss_list).mean(), global_step=global_step
            )
        train_loss_list.append(
            np.array([l.cpu().detach().numpy() for l in loss_list]).mean()
        )
        global_step += 1

    return train_loss_list, model, label_distr_shift, global_step, optimizer