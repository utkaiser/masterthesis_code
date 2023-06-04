from models.train_end_to_end import train_Dt_end_to_end


def component_grid_search_end_to_end(
        experiment_index = 0
):
    '''
    experiment_index: (int) number of the experiment, explained in paper

    Returns
    -------
    performs grid search to run end-to-end model on different component configurations
    '''

    model_resolutions = [
        128,
        # 256
    ]


    if experiment_index == 0:
        # EXPERIMENT 0: Test
        downsampling_models = ["Interpolation"]
        upsampling_models = ["UNet3"]

    else:
        # EXPERIMENT 1
        downsampling_models = [
            "Interpolation",
            "CNN"
        ]
        upsampling_models = [
            "UNet3",
            "UNet6",
            "Tiramisu",
            "UTransform",
            "Numerical_upsampling"
        ]

    # EXPERIMENT 3
    if experiment_index == 3:
        flipping = True
    else:
        flipping = False

    # EXPERIMENT 4
    if experiment_index == 4:
        multi_step = True
    else:
        multi_step = False

    # EXPERIMENT 5
    if experiment_index == 5:
        weighted_loss = True
        multi_step = True
    else:
        weighted_loss = False


    for d in downsampling_models:
        for u in upsampling_models:
            for res in model_resolutions:
                train_Dt_end_to_end(
                    downsampling_model = d,
                    upsampling_model = u,
                    model_res = res,
                    flipping = flipping,
                    multi_step = multi_step,
                    experiment_index = experiment_index,
                    weighted_loss = weighted_loss
                )


if __name__ == "__main__":
    component_grid_search_end_to_end()
