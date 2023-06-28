import numpy as np
import torch
from scipy.io import loadmat
from skimage.filters import gaussian


def get_velocities(res_padded, velocity_profiles, optimization):
    """
    Parameters
    ----------
    res_padded : (int) padded resolution; space added to actual velocity profile in case of parareal and absorbing bc
                        when using pseudo-spectral
    velocity_profiles : (string) name of velocity profiles; "bp_marmousi" or "mixed"
    optimization : (string) optimization technique; "parareal" or "none"
    boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"

    Returns
    -------
    get velocity profiles tensor (numpy)
    """
    input_path = f"../data/velocity_profiles/crops_bp_m_400_{res_padded}.npz"

    if velocity_profiles == "bp_marmousi":
        return np.load(input_path)["wavespeedlist"]

    else:  # velocity_profiles == "mixed"
        percentage_rest = 84  # assumption 400 different velocity profiles bp_m
        # velocity_accumulated = np.load(input_path)['wavespeedlist']

        counter = 0
        for vel_name in ["diagonal", "three_layers", "crack_profile", "wave_guide"]:
            crop = get_velocity_crop(
                res_padded, percentage_rest, vel_name, optimization
            )
            if counter == 0:
                counter += 1
                velocity_accumulated = crop
            else:
                velocity_accumulated = np.concatenate(
                    (velocity_accumulated, crop), axis=0
                )
        return velocity_accumulated


def get_velocity_crop(resolution, n_crops, velocity_profile, optimization):
    """
    Parameters
    ----------
    resolution : (int) resolution of crop
    n_crops : (int) number of crops
    velocity_profile : (string) name of velocity profiles; "bp_marmousi" or "mixed"
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    get velocity crops with special, crafted features:
    "diagonal", "marmousi", "marmousi2", "bp", "three_layers", "crack_profile", "high_frequency"
    """

    if velocity_profile == "diagonal":
        img = diagonal_ray(n_crops, resolution, optimization)

    elif velocity_profile == "marmousi":
        marmousi_datamat = loadmat(
            "../data/velocity_profiles/marm1nonsmooth.mat"
        )  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat["marm1larg"], 4)  # to make smoother
        img = np.expand_dims(
            marmousi_img[200 : 200 + resolution, 200 : 200 + resolution], axis=0
        )

    elif velocity_profile == "marmousi2":
        marmousi_datamat = loadmat(
            "../data/velocity_profiles/marm1nonsmooth.mat"
        )  # velocity models Marmousi dataset
        marmousi_img = gaussian(marmousi_datamat["marm1larg"], 4)  # to make smoother
        img = np.expand_dims(
            marmousi_img[300 : 300 + resolution, 300 : 300 + resolution], axis=0
        )

    elif velocity_profile == "bp":
        databp = loadmat(
            "../data/velocity_profiles/bp2004.mat"
        )  # velocity models BP dataset
        img = (
            gaussian(databp["V"], 4) / 1000
        )  # to make smoother (and different order of magnitude)
        img = np.expand_dims(
            img[1100 : 1100 + resolution, 1100 : 1100 + resolution], axis=0
        )

    elif velocity_profile == "three_layers":
        img = three_layers(n_crops, resolution, optimization)

    elif velocity_profile == "crack_profile":
        img = crack_profile(n_crops, resolution, optimization)

    elif velocity_profile == "high_frequency":
        img = high_frequency(n_crops, resolution, optimization)

    elif velocity_profile == "wave_guide":
        img = wave_guide(n_crops, resolution, optimization)

    else:
        raise NotImplementedError("Velocity model not implemented.")

    return img


def diagonal_ray(n_it, res, optimization):
    """
    Parameters
    ----------
    n_it : (int) number of samples
    res : (int) resolution of crop
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    velocity profiles with a diagonal ray as seen in paper,
    used to analyze generalization ability of end-to-end model
    """

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if optimization == "none":
        vel_profile = torch.from_numpy(
            3.0 + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.0) > 0.3 / 2)
        )
    else:
        vel_profile = torch.from_numpy(
            3.0 + 0.0 * yy - 1.5 * (np.abs(yy + xx - 0.0) > 0.3 / 3)
        )

    return vel_profile.unsqueeze(dim=0).repeat(n_it, 1, 1).numpy()


def three_layers(n_it, res, optimization):
    """
    Parameters
    ----------
    n_it : (int) number of samples
    res : (int) resolution of crop
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    velocity profiles with three layers as seen in paper,
    used to analyze generalization ability of end-to-end model
    """

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if optimization == "none":
        vel_profile = torch.from_numpy(
            2.6
            + 0.0 * yy
            - 0.7 * (yy + xx - 0.0 > -0.4 / 2)
            - 0.7 * (yy + xx - 0.0 > 0.6 / 2)
        )
    else:
        vel_profile = torch.from_numpy(
            2.6
            + 0.0 * yy
            - 0.7 * (yy + xx - 0.0 > -0.4 / 3)
            - 0.7 * (yy + xx - 0.0 > 0.6 / 3)
        )
    return vel_profile.unsqueeze(dim=0).repeat(n_it, 1, 1).numpy()


def crack_profile(n_it, res, optimization):
    """
    Parameters
    ----------
    n_it : (int) number of samples
    res : (int) resolution of crop
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    velocity profiles with cracks in marmousi profile as seen in paper,
    used to analyze generalization ability of end-to-end model
    """

    marmousi_datamat = loadmat(
        "../data/velocity_profiles/marm1nonsmooth.mat"
    )  # velocity models Marmousi dataset
    img = gaussian(marmousi_datamat["marm1larg"], 4)

    k1, k2, k3, k4 = 0.25, 0.25, 0.25, 0.25

    if optimization == "none":
        if res == 128 * 2:
            offset = 128 // 2
            vel_profile = img[1100 : 1100 + res, 1100 : 1100 + res]
            vel_profile[50 + offset : 70 + offset, 97 + offset : 123 + offset] = k1
            vel_profile[10 + offset : 28 + offset, 22 + offset : 31 + offset] = k2
            vel_profile[60 + offset : 118 + offset, 10 + offset : 28 + offset] = k3
            vel_profile[100 + offset : 118 + offset, 60 + offset : 80 + offset] = k4

        else:  # res == 256 * 2:
            offset = 256 // 2
            vel_profile = img[900 : 900 + res, 900 : 900 + res]
            vel_profile[100 + offset : 137 + offset, 200 + offset : 245 + offset] = k1
            vel_profile[18 + offset : 60 + offset, 37 + offset : 60 + offset] = k2
            vel_profile[120 + offset : 240 + offset, 20 + offset : 60 + offset] = k3
            vel_profile[195 + offset : 230 + offset, 120 + offset : 160 + offset] = k4
    else:
        if res == 128 * 3:
            offset = 128
            vel_profile = img[1100 : 1100 + res, 1100 : 1100 + res]
            vel_profile[50 + offset : 70 + offset, 97 + offset : 123 + offset] = k1
            vel_profile[10 + offset : 28 + offset, 22 + offset : 31 + offset] = k2
            vel_profile[60 + offset : 118 + offset, 10 + offset : 28 + offset] = k3
            vel_profile[100 + offset : 118 + offset, 60 + offset : 80 + offset] = k4
        else:
            raise NotImplementedError(
                "Resolution 256 x 256 for optimization and absorbing bc not implemented."
            )

    return torch.from_numpy(vel_profile).unsqueeze(dim=0).repeat(n_it, 1, 1).numpy()


def high_frequency(n_it, res, optimization):
    """
    Parameters
    ----------
    n_it : (int) number of samples
    res : (int) resolution of crop
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    velocity profiles with high frequency as seen in paper,
    used to analyze generalization ability of end-to-end model
    """

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if optimization == "none":
        factor = 2
    else:
        factor = 3

    vel_profile = torch.from_numpy(1.0 + 0.0 * yy)
    k = 0.03 / factor

    for i in range(res):
        if i < res // 2:
            vel_profile[i:, i:] += k
        else:
            vel_profile[i:, i:] -= k

    return vel_profile.unsqueeze(dim=0).repeat(n_it, 1, 1).numpy()


def wave_guide(n_it, res, optimization):
    """
    Parameters
    ----------
    n_it : (int) number of samples
    res : (int) resolution of crop
    optimization : (string) optimization technique; "parareal" or "none"

    Returns
    -------
    velocity profiles with wave guide implementation as seen in paper,
    used to analyze generalization ability of end-to-end model
    """

    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    if optimization == "none":
        vel_profile = torch.from_numpy(1 - 0.3 * np.cos(np.pi * xx)) * 3
    else:
        vel_profile = torch.from_numpy(1 - 0.3 * np.cos(np.pi * xx)) * 3

    return vel_profile.unsqueeze(dim=0).repeat(n_it, 1, 1).numpy()
