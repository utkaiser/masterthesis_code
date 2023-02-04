import torch
import numpy as np

if __name__ == '__main__':
    np_array = np.load("../data/end_to_end_0diag__3l__cp__hf__bp_m128_parareal.npz")
    a = torch.stack((torch.from_numpy(np_array['Ux']),
                     torch.from_numpy(np_array['Uy']),
                     torch.from_numpy(np_array['Utc']),
                     torch.from_numpy(np_array['vel'])), dim=2)
    print(a.shape)





