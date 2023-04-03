from models.train_end_to_end import train_Dt_end_to_end


def grid_search_end_to_end():

    def apply_rules(scale, res, counter):
        rules_bool = ((scale == 4 and res == 256) or (scale == 2 and res == 128)) \
                     and counter >= -1
        return rules_bool

    downsampling_model = [
        # "Interpolation",
        "CNN",
        # "Simple"
    ]
    upsampling_model = [
        "UNet3",
        # "UNet6",
        # "Tiramisu",
        # "UTransform",
        # "Numerical_upsampling"
    ]
    optimizer = [
        "AdamW",
        # "RMSprop",
        # "SGD"
    ]
    loss_function = [
        "SmoothL1Loss",
        # "MSE"
    ]
    res_scaler = [
        2,
        # 4
    ]
    model_res = [
        128,
        # 256,
    ]
    flipping = [
        False,
        # True
    ]
    multi_step = [
        # 1,
        # 2,
        -1  # shifting normal distribution
    ]

    counter = 0
    for d in downsampling_model:
        for u in upsampling_model:
            for o in optimizer:
                for l in loss_function:
                    for scale in res_scaler:
                        for res in model_res:
                                for f in flipping:
                                    for m in multi_step:
                                        if apply_rules(scale, res, counter):
                                            train_Dt_end_to_end(
                                                downsampling_model = d,
                                                upsampling_model = u,
                                                optimizer_name = o,
                                                loss_function_name = l,
                                                res_scaler = scale,
                                                model_res = res,
                                                flipping = f,
                                                multi_step = m
                                            )
                                        counter += 1


if __name__ == "__main__":
    grid_search_end_to_end()
