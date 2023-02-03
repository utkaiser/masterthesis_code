import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from generate_data.optimization.utils_optimization import compute_loss


def plot_heatmap_optimization(fine_solver_tensor, parareal_tensor, vel, vel_name, folder_name):
    # coarse_solver_tensor -> s x b x c x w x h
    # fine_solver_tensor -> s x b x c x w x h
    # parareal_tensor -> k x s x b x c x w x h

    parareal_vs_fine_errors = []
    for k in range(parareal_tensor.shape[0]):
        curr_errors = []
        for s in range(1,parareal_tensor.shape[1]):
            curr_errors.append(compute_loss(parareal_tensor[k,s], fine_solver_tensor[s], vel))
        parareal_vs_fine_errors.append(curr_errors)

    _ = sns.heatmap(np.array(parareal_vs_fine_errors))
    plt.title("MSE between end-to-end and fine solver in energy components.")
    plt.xlabel("time step")
    plt.ylabel("parareal iteration")
    plt.savefig('../../results/optimization/'+folder_name+'/' + vel_name + '_heatmap.pdf')
    plt.close()