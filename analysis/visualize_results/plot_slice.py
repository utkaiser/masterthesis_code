import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from generate_data.initial_conditions import diagonal_ray
import torch

from generate_data.optimization.utils_optimization import get_wavefield

sns.set_context()
sns.set_theme(style='white', context="paper")

def compare_slice():

    parareal_it = 1

    tensors = {
        "coarse solver": torch.from_numpy(np.load(
            '../../results/optimization/first_tries/check_stability/diagonal_coarse.npy')),  # s x b x c x w x h
        "fine solver": torch.from_numpy(np.load(
            '../../results/optimization/first_tries/check_stability/diagonal_fine.npy'))  # s x b x c x w x h
    }
    parareal_tensor = torch.from_numpy(np.load(
        '../../results/optimization/first_tries/check_stability/diagonal_parareal.npy'))  # k x s x b x c x w x h
    tensors["parareal it " + str(parareal_it)] = parareal_tensor[parareal_it]
    tensors["network"] = parareal_tensor[0]

    resolution = tensors["coarse solver"].shape[-1]
    vel = torch.from_numpy(diagonal_ray(1, res=resolution)).squeeze()

    plot_style = {
        "coarse solver": {"style": "dashed", "color": "green"},
        "fine solver": {"style": "solid", "color": "blue"},
        "parareal it " + str(parareal_it): {"style": "dashdot", "color": "orange"},
        "network": {"style": "dotted", "color": "red"}
    }

    fig = plt.figure(figsize=(15, 17))
    i, time_steps = 1, [3,4,5]
    for s in time_steps:
        _ = fig.add_subplot(4, 1, i)

        for name, tensor in tensors.items():
            img = tensor[s]  # b x c x w x h
            wavefield = get_wavefield(img,vel)  # w x h
            wavefield_row = get_row(wavefield, row_number = 40)  # w
            plt.plot(wavefield_row, linestyle=plot_style[name]["style"],
                     linewidth=.7, color=plot_style[name]["color"])

        if i == 1: plt.legend(tensors.keys(), prop={'size': 13})
        if s == time_steps[-1]:
            plt.xlabel("x")
            plt.ylabel("wave field")
        plt.title("T = "+str(s)+"*.06")

        i += 1

    plt.savefig("../results/parareal/slice_plots/slice_plot_parareal1_a.pdf")
    plt.close()



def get_row(wavefield, row_number):
    # wavefield -> w x h
    return wavefield[:,row_number]



if __name__ == '__main__':
    compare_slice()