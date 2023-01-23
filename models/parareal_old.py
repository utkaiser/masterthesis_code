import torch
from models.parallel_scheme import one_iteration_pseudo_spectral, smaller_crop


def parareal_procrustes_scheme(model, u_0, _, n_parareal = 4, n_snapshots = 5):

    # u_0 -> b x c x w x h

    u_n = u_0.clone()
    vel = u_n[:,3].clone().unsqueeze(dim=0)  # 1 x 1 x 500 x 500
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 500 x 500
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, 128, 128])
    big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])

    # initial guess, first iteration without parareal
    parareal_tensor[0, 0] = smaller_crop(u_n[:, :3].clone())
    big_tensor[0] = u_0[:, :3].clone()
    for n in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 512 x 512
        parareal_tensor[0,n+1] = smaller_crop(u_n1)
        big_tensor[n+1] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # parareal iterations: k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):
        print(k)

        parareal_tensor[k, 0] = smaller_crop(u_0[:, :3].clone())
        res_fine, Un, Vn, res_model = get_optimizing_terms(model, big_tensor, n_snapshots, vel)  # n_snapshots x b x c x w x h
        new_big_tensor = torch.zeros([n_snapshots, batch_size, channel - 1, width, height])
        new_big_tensor[0] = u_0[:, :3].clone()

        for n in range(n_snapshots-1):
            u_n_k1 = torch.cat((new_big_tensor[n], vel), dim=1)
            u_n1_k1 = apply_procrustes(model(u_n_k1).squeeze(),res_fine[n],res_model[n],Un,Vn)
            parareal_tensor[k, n+1] = smaller_crop(u_n1_k1)
            new_big_tensor[n+1] = u_n1_k1

        big_tensor = new_big_tensor.clone()

    return parareal_tensor  # k x s x b x c x w x h


def get_optimizing_terms(model, big_pseudo_tensor, _, vel):
    # this can be done later computed in parallel

    n_snapshots, b, c, w, h = big_pseudo_tensor.shape
    res_fine = torch.zeros([n_snapshots, c, w, h]).double()
    res_model = torch.zeros([n_snapshots, c, w, h]).double()  # n_snapshots x c x w x h

    for s in range(n_snapshots):
        res_fine[s], res_model[s] = compute_parareal_term(model, torch.cat([big_pseudo_tensor[s], vel], dim=1))

    Un, Vn = compute_procrustes(res_fine, res_model)

    return res_fine, Un, Vn, res_model


def compute_parareal_term(model, u_n_k):
    # u_n_k -> b x c x w x h

    res_model = model(u_n_k)  # 1 x 3 x w x h
    res_fine_solver = one_iteration_pseudo_spectral(u_n_k)  # 1 x 3 x w x h

    return res_fine_solver.squeeze(), res_model.squeeze()


def compute_procrustes(m, t):
    m, t = reshape_channel_wise(m), reshape_channel_wise(t)

    QA, RA = torch.linalg.qr(m, mode='reduced')
    QB, RB = torch.linalg.qr(t, mode='reduced')
    up, sp, vtp = torch.linalg.svd(torch.matmul(RA, RB.transpose(0,1)))
    Un = torch.matmul(QA, up)
    Vn = torch.matmul(QB, vtp.transpose(0,1))

    print('Coarse error:', torch.linalg.norm(m - t, ord='fro'),
          '| OPP error:', torch.linalg.norm(m - torch.matmul(Un, torch.matmul(Vn.transpose(0,1), t)), ord='fro'))

    return Un,Vn


def reshape_channel_wise(matrix):
    n_snapshots, channel, w, h = matrix.shape
    new_matrix = torch.zeros([w * h * channel, n_snapshots])
    for s in range(n_snapshots):
        new_matrix[:, s] = reshape(matrix[s])
    return new_matrix


def reshape(matrix):
    a = matrix[0].view(-1)
    b = matrix[1].view(-1)
    c = matrix[2].view(-1)
    return torch.cat([a, b, c], dim=0)


def apply_procrustes(res_model, res_fine, res_model_prev, U,V):
    # res_model -> 3 x 256 x 256

    res_model_flat = reshape(res_model)
    res_model_prev_flat = reshape(res_model_prev)

    r = torch.matmul(V.transpose(0,1).double(), res_model_flat.double())

    res_model_flat_after = torch.matmul(U.double(),r)
    res_model_prev_flat = torch.matmul(U.double(), torch.matmul(V.transpose(0,1).double(), res_model_prev_flat.double()))

    print("i", res_model_flat.shape, res_model_flat_after.shape, r.shape, V.shape)

    c, w, h = res_model.shape

    return reshape_back(res_model_flat_after,c, w, h) + (res_fine - reshape_back(res_model_prev_flat,c, w, h))


def reshape_back(matrix, channel, w, h):
    # matrix -> 196608, 1

    new_matrix = torch.zeros([channel, w, h])

    new_matrix[0] = torch.reshape(matrix[:w*h*1],(w,h))
    new_matrix[1] = torch.reshape(matrix[w*h*1:w * h * 2],(w,h))
    new_matrix[2] = torch.reshape(matrix[w*h*2:w * h * 3],(w,h))

    return new_matrix



