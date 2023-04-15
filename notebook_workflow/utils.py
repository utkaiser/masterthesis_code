import numpy as np
import matplotlib.pyplot as plt
import torch

from generate_data.change_wave_arguments import WaveSol_from_EnergyComponent_tensor, WaveEnergyField_tensor
from generate_data.utils_wave_propagate import one_iteration_pseudo_spectral_tensor


def get_velocity_model(
        data_path,
        visualize = True
):
    '''

    Parameters
    ----------
    data_path : (string) path to velocity profile crops
    visualize : (boolean) whether to visualize data

    Returns
    -------
    (numpy array) single velocity profile
    '''

    # choose first velocity profile out of list of velocity crops
    vel = np.load(data_path)['wavespeedlist'].squeeze()[0]

    if visualize:
        plt.axis("off")
        plt.title("Velocity profile")
        plt.imshow(vel)
        plt.show()

    return vel


def pseudo_spectral_solutions(
        u_energy,
        vel,
        n_it,
        dx,
        dt,
        dt_star
):

    print(f"Solve wave equation using pseudo-spectral method.")

    b, n_comp, w, h = u_energy.shape
    target = torch.zeros([n_it, n_comp, w, h])

    for i in range(n_it):
        u_energy = one_iteration_pseudo_spectral_tensor(
            u_n_k = torch.cat([u_energy, torch.from_numpy(vel).unsqueeze(dim=0).unsqueeze(dim=0)], dim=1),
            f_delta_x = dx,
            f_delta_t = dt,
            delta_t_star = dt_star
        )
        target[i] = u_energy.clone()

    return target


def visualize_parareal(
        pseudo_spectral_tensor,
        parareal_tensor,
        n_parareal,
        n_it,
        f_delta_x,
        vel
):
    fig = plt.figure(figsize=(30, 10))

    for s in range(n_it):

        ax = fig.add_subplot(n_parareal + 1, n_it, 1 + s)
        ax.set_title(f"PS propagation {s}")

        u_energy = pseudo_spectral_tensor[s]
        u, ut = WaveSol_from_EnergyComponent_tensor(u_energy[0].unsqueeze(dim=0),
                                                    u_energy[1].unsqueeze(dim=0),
                                                    u_energy[2].unsqueeze(dim=0),
                                                    torch.from_numpy(vel).unsqueeze(dim=0),
                                                    f_delta_x,
                                                    torch.sum(torch.sum(u_energy[0])))
        w = WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), torch.from_numpy(vel), f_delta_x)

        ax.imshow(w)

    for s in range(n_it):
        for k in range(n_parareal):

            ax = fig.add_subplot(n_parareal + 1, n_it, 11 + 10 * k + s)
            ax.set_title(f"Parareal it {k}, prop. {s}")

            u_energy = parareal_tensor[k, s]
            u, ut = WaveSol_from_EnergyComponent_tensor(u_energy[0, 0].unsqueeze(dim=0),
                                                        u_energy[0, 1].unsqueeze(dim=0),
                                                        u_energy[0, 2].unsqueeze(dim=0),
                                                        torch.from_numpy(vel).unsqueeze(dim=0),
                                                        f_delta_x,
                                                        torch.sum(torch.sum(u_energy[0, 0])))
            w = WaveEnergyField_tensor(u.squeeze(), ut.squeeze(), torch.from_numpy(vel), f_delta_x)

            ax.imshow(w)

    plt.show()

