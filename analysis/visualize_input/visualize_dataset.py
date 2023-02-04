import sys
import torch
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler
from generate_data.optimization_generate_data.utils_optimization import get_wavefield
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
sys.path.append("..")
from generate_data.initial_conditions import get_velocity_dict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

def vis_velocities():

    input_path = "../../data/velocity_profiles/crops_bp_m_200_128.npz"
    velocities = get_velocity_dict(128,10,input_path)

    # choose how many from which
    for key, value in velocities.items():
        if key == "bp_m":
            velocities[key] = np.concatenate([value[:8],value[-8:]], axis=0)
        else:
            velocities[key] = np.expand_dims(value[0], axis=0)

    velocity_tensor = np.concatenate(list(velocities.values()), axis=0)
    print(velocity_tensor.shape)

    fig = plt.figure(figsize=(8, 8))

    for i in range(velocity_tensor.shape[0]):
        vel = velocity_tensor[i]
        a = fig.add_subplot(5, 4, i+1)
        plt.imshow(vel)
        a.set_aspect('equal')
        plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=-.4, hspace=.1)
    plt.savefig("visualized_velocity_profiles.pdf")
    plt.close(fig)


def pca_dataset_save_mat():

    _ = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")

    data_path = "../../data/end_to_end_0diag__3l__cp__hf__bp_m128.npz"

    np_array = np.load(data_path)
    tensor = torch.stack((torch.from_numpy(np_array['Ux']),
                 torch.from_numpy(np_array['Uy']),
                 torch.from_numpy(np_array['Utc']),
                 torch.from_numpy(np_array['vel'])), dim=2)
    # tensor -> :50 - diag, 50:100 - 3l, 100:150 - cp, 150:200 - hf, 200:400 - bp, 400: - m
    tensor = torch.concat([tensor[:10],tensor[50:60], tensor[100:110],tensor[150:160],tensor[200:210], tensor[400:410]],dim=0)

    counter = 0
    n_samples, n_snaps, c, w, h = tensor.shape
    new_tensor = torch.zeros([n_samples*n_snaps,w*h])
    for n in range(n_samples):
        for s in range(n_snaps):
            wave = tensor[n,s,:3].unsqueeze(dim=0)
            vel = tensor[n,s,3].unsqueeze(dim=0)
            wavefield = get_wavefield(wave,vel)
            new_tensor[counter] = wavefield.view(-1)
            counter += 1

    # do pca with 3 orthonormal vectors
    pca = PCA(3)
    pipe = Pipeline([('scale', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(new_tensor)

    savemat("../../results/datagen/pca/pca3_scaled.mat", {"res": Xt})

    # labels = {
    #     "diagonal": ["red", 100],
    #     "3 layers": ["blue", 200],
    #     "cracks": ["green", 300],
    #     "high frequency": ["orange", 400],
    #     "BP": ["magenta", 500],
    #     "Marmousi": ["black", 600]
    # }
    #
    # prev = 0
    # for v, indx in labels.values():
    #     ax.scatter(Xt[prev:indx, 0], Xt[prev:indx, 1], c=v)
    #     prev = indx
    #
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.set_zticklabels([])
    # plt.legend(labels.keys())
    # plt.show()

def visualize_dataset():
    np_array = np.load("../../data/val/end_to_end_val_3l_128.npz")
    tensor = torch.stack((
        torch.from_numpy(np_array['Ux']),
        torch.from_numpy(np_array['Uy']),
        torch.from_numpy(np_array['Utc']),
        torch.from_numpy(np_array['vel'])),
        dim=2)  # n_it x n_snaps x c x w x h

    vel = tensor[0,0,3].unsqueeze(dim=0)
    for i in range(tensor.shape[0]):
        for s in range(tensor.shape[1]):
            plt.imshow(get_wavefield(tensor[i,s].unsqueeze(dim=0), vel))
            plt.show()


if __name__ == '__main__':
    visualize_dataset()



