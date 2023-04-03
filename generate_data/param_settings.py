def get_training_params(res):

    param_dict = {
        "total_time": .6,
        "delta_t_star": .06,
        "n_snaps": 7
    }
    # param_dict["f_delta_x"], param_dict["f_delta_t"]

    if res == 128:
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.

    # TODO: test this

    else:  # res == 256
        param_dict["f_delta_x"] = 2.0 / 128.0
        param_dict["f_delta_t"] = param_dict["f_delta_x"] / 20.

    return param_dict