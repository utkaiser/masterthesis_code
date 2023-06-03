def get_training_params(
        res
):
    '''
    Parameters
    ----------
    res : (int) resolution of velocity profile

    Returns
    -------
    parameters used for generating data stored in dictionary
    '''

    param_dict = {
        "delta_t_star": .06,
        "n_snaps": 8
    }

    if res == 128:
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.

    else:  # res == 256
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.

    return param_dict