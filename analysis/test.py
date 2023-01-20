import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import orthogonal_procrustes
import torch


if __name__ == '__main__':

    # For reproducibility, use a seeded RNG.

    A, B = torch.rand(3,3), torch.rand(3,3)

    # Find Q.  Note that `orthogonal_procrustes` does not include
    # dilation or translation.  To handle translation, we center
    # A and B by subtracting the means of the points.
    A0 = A - A.mean()
    B0 = B - B.mean()

    Q, scale = orthogonal_procrustes(A0, B0)
    Q2, scale2 = orthogonal_procrustes(A, B)

    with np.printoptions(precision=3, suppress=True):
        print((A0 @ Q) + B.mean(axis=0, keepdims=True))
        print(B)
        print("-"*20)
        print(A @ Q2)
        print(B)
        print(torch.linalg.norm(((A0 @ Q)+ B.mean()) - B, ord="fro"))
        print(torch.linalg.norm((A @ Q2) - B, ord="fro"))






