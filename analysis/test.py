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
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)

    Q, scale = orthogonal_procrustes(A0, B0)

    with np.printoptions(precision=3, suppress=True):
        print(A0 @ Q)
        print(B0)








