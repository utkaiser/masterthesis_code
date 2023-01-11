import torch
from models import model_end_to_end
from generate_data.wave_propagation import pseudo_spectral
from generate_data.wave_util import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
import matplotlib.pyplot as plt
from models.model_utils import fetch_data_end_to_end, get_params


def parareal_scheme(model, u_n, n_parareal = 4, n_snapshots = 11):

    vel = u_n[:,3,:,:].clone().unsqueeze(dim=1)
    batch_size, channel, width, height = u_n.shape  # 1 x 4 x 128 x 128
    parareal_tensor = torch.zeros([n_parareal+1, n_snapshots, batch_size, channel-1, width, height])

    s = 0
    for k in range(n_parareal):
        parareal_tensor[k,s,:,:,:,:] = u_n[:,3,:,:].clone()

    k = 0
    for s in range(n_snapshots-1):
        u_n1 = model(u_n)  # 1 x 3 x 128 x 128
        parareal_tensor[k,s+1,:,:,:,:] = u_n1
        u_n = torch.cat((u_n1, vel), dim=1)

    # k = 1, 2, 3, 4
    for k in range(1,n_parareal+1):

        parareal_terms = get_parareal_terms(model, parareal_tensor[k-1], n_snapshots, vel) # n_snapshots x b x c x w x h

        for s in range(n_snapshots-1):
            u_n_k1 = torch.cat((parareal_tensor[k,s,:,:,:,:], vel), dim=1)
            u_n1_k1 = model(u_n_k1) + parareal_terms[s]
            parareal_tensor[k, s+1, :, :, :, :] = u_n1_k1

    return parareal_tensor  # k x s x b x c x w x h


def get_parareal_terms(model, parareal_tensor, n_snapshots, vel):
    # this can be later computed in parallel

    parareal_terms = torch.zeros(parareal_tensor.shape)
    for s in range(n_snapshots):
        parareal_terms[s] = compute_parareal_term(model, torch.cat([parareal_tensor[s], vel], dim=1))

    return parareal_terms


def compute_parareal_term(model, u_n1_k):

    # u_n_k -> b x c x w x h

    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    delta_t_star = .06

    u, u_t = WaveSol_from_EnergyComponent_tensor(u_n1_k[:, 0, :, :].clone(),
                                                   u_n1_k[:, 1, :, :].clone(),
                                                   u_n1_k[:, 2, :, :].clone(),
                                                   u_n1_k[:, 3, :, :].clone(),
                                                   f_delta_x,
                                                   torch.sum(torch.sum(torch.sum(u_n1_k[:, 0, :, :].clone()))))
    u, u_t, vel = u.squeeze().numpy(), u_t.squeeze().numpy(), u_n1_k[:, 3, :, :].clone().squeeze().numpy()
    u_prop, u_t_prop = pseudo_spectral(u, u_t, vel,f_delta_x, f_delta_t, delta_t_star)
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(torch.from_numpy(u_prop).unsqueeze(dim=0), torch.from_numpy(u_t_prop).unsqueeze(dim=0), torch.from_numpy(vel).unsqueeze(dim=0), f_delta_x)
    res_fine_solver = torch.stack([u_x, u_y, u_t_c], dim=1)

    return res_fine_solver - model(u_n1_k)


def vis_parareal():

    # data
    f_delta_x = 2.0 / 128.0
    f_delta_t = f_delta_x / 20
    param_dict = get_params("0")
    path = ['../data/end_to_end_bp_m_10_2000.npz']
    loader, _ = fetch_data_end_to_end(path, val_paths=path, shuffle=True, batch_size=1)

    # param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_end_to_end.Restriction_nn(param_dict=param_dict).double().to(device)
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('../results/run_2/good_one/saved_model_end_to_end_only_unet3lvl128_10.pt'))
    # model.eval()

    fig = plt.figure(figsize=(35, 8))

    with torch.no_grad():
        for i, data in enumerate(loader):

            inpt = data[0].squeeze()  # n_snaps x 4 x w x h
            curr_inpt = inpt[0, :, :, :].unsqueeze(dim=0)  # b x 4 x w x h
            u_x, u_y, u_t_c, vel = curr_inpt[:, 0, :, :], curr_inpt[:, 1, :, :], curr_inpt[:, 2, :, :], curr_inpt[:, 3, :, :]
            u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t,
                                                         torch.sum(torch.sum(torch.sum(u_x))))

            ax0 = fig.add_subplot(1, 8, 1)
            pos0 = ax0.imshow(WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(), f_delta_x) * f_delta_x * f_delta_x)
            plt.colorbar(pos0)

            # # TODO: change to vv
            # for s in range(1,inpt.shape[0]-4):
            #     output = model(curr_inpt) # b x 4 x w x h
            #     u_x, u_y, u_t_c = output[:,0,:,:], output[:,1,:,:], output[:,2,:,:]
            #     u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, torch.sum(torch.sum(torch.sum(u_x))))
            #     ax = fig.add_subplot(5, 8, s+1)
            #     pos = ax.imshow(WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(), f_delta_x) * f_delta_x * f_delta_x)
            #     plt.colorbar(pos)
            #     plt.axis('off')
            #     curr_inpt = torch.cat([output, vel.unsqueeze(dim=1)], dim=1)



            # parareal scheme
            u_n = inpt[0, :, :, :].unsqueeze(dim=0)
            parareal_tensor = parareal_scheme(model, u_n)  # k x s x b x c x w x h

            fig = plt.figure(figsize=(35, 8))
            for k in range(parareal_tensor.shape[0]):
                for s in range(parareal_tensor.shape[1]):
                    wave_field = get_wavefield(parareal_tensor[k,s,:,:,:,:], vel)

                    ax = fig.add_subplot(5, 11, 11*k + s + 1)
                    pos = ax.imshow(wave_field)
                    plt.colorbar(pos)
                    plt.axis('off')


            plt.show()
            break



def get_wavefield(tensor, vel, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20):

    u_x, u_y, u_t_c = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]
    u, u_t = WaveSol_from_EnergyComponent_tensor(u_x, u_y, u_t_c, vel, f_delta_t, torch.sum(torch.sum(torch.sum(u_x))))
    return WaveEnergyField_tensor(u.squeeze().cpu(), u_t.squeeze().cpu(), vel.squeeze().cpu(),
                                           f_delta_x) * f_delta_x * f_delta_x




if __name__ == '__main__':
    vis_parareal()





