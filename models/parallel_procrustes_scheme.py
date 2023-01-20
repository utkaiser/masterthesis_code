import scipy
import torch
from models.parallel_scheme import one_iteration_pseudo_spectral, smaller_crop


def parareal_procrustes_scheme(model, u_0, _, n_parareal = 4, n_snapshots = 11):

    # u_0 -> b x c x w x h

    u_n = u_0.clone()
    vel = u_n[:,3].clone().unsqueeze(dim=1)  # 1 x 1 x 500 x 500
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 500 x 500
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, 128, 128])
    big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

    # initial guess, first iteration without parareal
    parareal_tensor[0, 0] = smaller_crop(u_n[:, :3].clone())
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 512 x 512
        parareal_tensor[0,n+1] = smaller_crop(u_n1)
        big_tensor[n+1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print(k)

        big_tensor[0] = u_0[:, :3].clone()
        parareal_tensor[k, 0] = smaller_crop(u_0[:, :3].clone())
        res_fine, procrustes_matrices, res_model = get_optimizing_terms(model, big_tensor, n_snapshots, vel)  # n_snapshots x b x c x w x h
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
        new_big_tensor[0] = u_0[:, :3].clone()

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((new_big_tensor[n], vel), dim=1)
            R = procrustes_matrices[n+1]
            u_n1_k1 = torch.matmul(model(u_n_k1).squeeze(), R) + (res_fine[n] - torch.matmul(res_model[n],R))
            parareal_tensor[k, n+1] = smaller_crop(u_n1_k1)
            new_big_tensor[n+1] = u_n1_k1

        big_tensor = new_big_tensor.clone()

    return parareal_tensor  # k x s x b x c x w x h


def get_optimizing_terms(model, big_pseudo_tensor, _, vel):
    # this can be done later computed in parallel

    n_snapshots, b, c, w, h = big_pseudo_tensor.shape
    res_fine = torch.zeros([n_snapshots, c, w, h]).double()
    procrustes_matrices = torch.zeros([n_snapshots, c, w, h]).double()  # n_snapshots x c x w x h
    res_model = torch.zeros([n_snapshots, c, w, h]).double()  # n_snapshots x c x w x h

    for s in range(n_snapshots):
        res_fine[s], procrustes_matrices[s], res_model[s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[s], vel], dim=1))

    return res_fine, procrustes_matrices, res_model


def compute_parareal_term(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # 1 x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral(u_n_k)  # 1 x 3 x w x h
    procrustes_matrix = procrustes_optimization(res_model, res_fine_solver) # c, 2, w, h

    return res_fine_solver.squeeze(), procrustes_matrix.squeeze(), res_model.squeeze()


def procrustes_optimization(matrix, target):
    # matrix -> 1 x 3 x w x h
    b,c,w,h = matrix.shape
    procrustes_matrix = torch.zeros(b,c,w,h).double()

    # channel-wise procrustes
    for c in range(matrix.shape[1]):
        m, t = matrix[0,c], target[0,c]
        procrustes_matrix[0,c] = compute_procrustes(m, t)

    return procrustes_matrix


def compute_procrustes(m, t):

    R, _ = scipy.linalg.orthogonal_procrustes(m, t)

    return torch.from_numpy(R)





