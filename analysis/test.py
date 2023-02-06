import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    np_array = np.load("../data/D_t_128_parareal/end_to_end_0diag__3l__cp__hf__bp_m256_parareal.npz")
    a = torch.stack((torch.from_numpy(np_array['Ux']),
                     torch.from_numpy(np_array['Uy']),
                     torch.from_numpy(np_array['Utc']),
                     torch.from_numpy(np_array['vel'])), dim=2)

    for i in range(3, a.shape[0]):
        for s in range(a.shape[1]):
            plt.imshow(a[i,s,0,64:-64,64:-64])
            plt.show()





