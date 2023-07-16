import sys
import random

import numpy as np
import torch

from models.utils import sample_label_normal_dist
from parareal.parallel_scheme_training import parareal_scheme

sys.path.append("..")
sys.path.append("../..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_parareal(
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

            input_tensor = data[:, input_idx].detach()  # b x 4 x w x h

            label_tensor_all = data[:, input_idx + 1 : , :3].detach()
            loss_list += parareal_scheme(model, input_tensor, label_tensor_all, loss_f, 4, 8 - input_idx + 1)

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