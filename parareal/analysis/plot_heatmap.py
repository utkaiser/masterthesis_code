import sys
sys.path.append("")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_wavefield_heatmap(_, fine_solver_tensor, parareal_tensor, __, MSE_loss, ___, vel_name):
    # coarse_solver_tensor -> s x b x c x w x h
    # fine_solver_tensor -> s x b x c x w x h
    # parareal_tensor -> k x s x b x c x w x h

    parareal_vs_fine_errors = []
    for k in range(parareal_tensor.shape[0]):
        curr_errors = []
        for s in range(1,parareal_tensor.shape[1]):
            curr_errors.append(MSE_loss(parareal_tensor[k,s], fine_solver_tensor[s]))
        parareal_vs_fine_errors.append(curr_errors)

    _ = sns.heatmap(np.array(parareal_vs_fine_errors))
    plt.title("MSE between end-to-end and fine solver.")
    plt.xlabel("time step")
    plt.ylabel("parareal iteration")
    plt.savefig('../../results/parareal/check_stability/' + vel_name + '_heatmap.pdf')
    plt.close()