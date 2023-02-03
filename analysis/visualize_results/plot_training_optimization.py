import matplotlib.pyplot as plt
from analysis.utils_analysis import get_ticks_fine
from generate_data.optimization.utils_optimization import get_wavefield


def plot_big_tensor(big_tensor, vel, data):
    # big_tensor -> k+1 x b x n_snaps=2 x c-1 x w x h
    # vel -> b x 1 x 256 x 256
    # data -> b x s x c x w x h

    b = 0
    fig = plt.figure(figsize=(20, 5))
    ticks = get_ticks_fine(data[b,:,:3].unsqueeze(dim=0), vel[b,0])

    for k in range(big_tensor.shape[0]):
        for s in range(big_tensor.shape[2]):
            ax = fig.add_subplot(big_tensor.shape[0], big_tensor.shape[2], k * big_tensor.shape[2] + 1 + s)
            wave_field = get_wavefield(big_tensor[k,b,s].unsqueeze(dim=0), vel[b])
            pos = ax.imshow(wave_field, vmin=ticks[s][0], vmax=ticks[s][2])
            plt.colorbar(pos, ticks=ticks[s])
            ax.set_title(str(k) + ", " + str(s))
            plt.axis('off')

    plt.show()


