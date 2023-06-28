import sys

import torch
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid", {"axes.grid": False})
from utils import get_wavefield


def pca_dataset_save_mat():
    _ = plt.figure(figsize=(8, 6))
    __ = plt.axes(projection="3d")

    data_path = "../data/..."

    np_array = np.load(data_path)
    tensor = torch.stack(
        (
            torch.from_numpy(np_array["Ux"]),
            torch.from_numpy(np_array["Uy"]),
            torch.from_numpy(np_array["Utc"]),
            torch.from_numpy(np_array["vel"]),
        ),
        dim=2,
    )
    # tensor -> :50 - diag, 50:100 - 3l, 100:150 - cp, 150:200 - hf, 200:400 - bp, 400: - m
    tensor = torch.concat(
        [
            tensor[:10],
            tensor[50:60],
            tensor[100:110],
            tensor[150:160],
            tensor[200:210],
            tensor[400:410],
        ],
        dim=0,
    )

    counter = 0
    n_samples, n_snaps, c, w, h = tensor.shape
    new_tensor = torch.zeros([n_samples * n_snaps, w * h])
    for n in range(n_samples):
        for s in range(n_snaps):
            wave = tensor[n, s, :3].unsqueeze(dim=0)
            vel = tensor[n, s, 3].unsqueeze(dim=0)
            wavefield = get_wavefield(wave, vel)
            new_tensor[counter] = wavefield.view(-1)
            counter += 1

    # do pca with 3 orthonormal vectors
    pca = PCA(3)
    pipe = Pipeline([("scale", StandardScaler()), ("pca", pca)])
    Xt = pipe.fit_transform(new_tensor)

    savemat("../results/datagen/pca/pca3_scaled.mat", {"res": Xt})


def visualize_dataset():
    np_array = np.load("../data/val/end_to_end_val_3l_128.npz")
    tensor = torch.stack(
        (
            torch.from_numpy(np_array["Ux"]),
            torch.from_numpy(np_array["Uy"]),
            torch.from_numpy(np_array["Utc"]),
            torch.from_numpy(np_array["vel"]),
        ),
        dim=2,
    )  # n_it x n_snaps x c x w x h

    vel = tensor[0, 0, 3].unsqueeze(dim=0)
    for i in range(tensor.shape[0]):
        for s in range(tensor.shape[1]):
            plt.imshow(get_wavefield(tensor[i, s].unsqueeze(dim=0), vel))
            plt.show()


if __name__ == "__main__":
    visualize_dataset()
