import matplotlib.pyplot as plt
from analysis.utils_analysis import get_ticks_fine, get_solver_solution
from models.model_utils import get_wavefield


def plot_big_tensor(big_tensor, vel, data, b = 0):
    # big_tensor -> k+1 x b x n_snaps x c-1 x w x h
    # vel -> b x 1 x 256 x 256
    # data -> b x s x c x w x h

    fig = plt.figure(figsize=(20, 5))
    ticks = get_ticks_fine(data[b,:,:3].unsqueeze(dim=0), vel[b,0])
    u_0 = big_tensor[0,b,0].unsqueeze(dim=0)
    coarse_solver_tensor = get_solver_solution(u_0,
                                               big_tensor.shape[2],
                                               vel[b].unsqueeze(dim=0),
                                               solver="coarse")  # s x b x c x w x h

    # coarse solver iteration solution
    for s in range(big_tensor.shape[2]):
        ax = fig.add_subplot(big_tensor.shape[0] + 2, big_tensor.shape[2], 1 + s)
        wave_field = get_wavefield(coarse_solver_tensor[s,b].unsqueeze(dim=0), vel[b])
        pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
        plt.colorbar(pos, ticks=ticks[s])
        ax.set_title("coarse it " + str(s))
        plt.axis('off')

    # parareal iteration solution
    for k in range(big_tensor.shape[0]):
        for s in range(big_tensor.shape[2]):
            ax = fig.add_subplot(big_tensor.shape[0] + 2, big_tensor.shape[2], (k+1) * big_tensor.shape[2] + 1 + s)
            wave_field = get_wavefield(big_tensor[k,b,s].unsqueeze(dim=0), vel[b])
            pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
            plt.colorbar(pos, ticks=ticks[s])
            ax.set_title(str(k) + ", " + str(s))
            plt.axis('off')

    # fine solver solution
    for s in range(big_tensor.shape[2]):
        ax = fig.add_subplot(big_tensor.shape[0] + 2, big_tensor.shape[2], (big_tensor.shape[0] + 1) * big_tensor.shape[2] + 1 + s)
        wave_field = get_wavefield(data[b,s,:3].unsqueeze(dim=0), vel[b])
        pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
        plt.colorbar(pos, ticks=ticks[s])
        ax.set_title("fine it " + str(s))
        plt.axis('off')

    plt.show()


