from models.train_end_to_end import train_Dt_end_to_end


def component_grid_search_end_to_end(
):
    '''
    void

    Returns
    -------
    performs grid search to run end-to-end model on different component configurations
    '''

    downsampling_models = [
        "Interpolation",
        #"CNN"
    ]

    upsampling_models = [
        "UNet3",
        # "UNet6",
        # "Tiramisu",
        # "UTransform",
        # "Numerical_upsampling"
    ]

    model_resolutions = [
        128,
        #256
    ]

    flipping = [
        False,
        #True
    ]

    multi_step = [
        False,
        # True
    ]

    for d in downsampling_models:
        for u in upsampling_models:
            for res in model_resolutions:
                for f in flipping:
                    for m in multi_step:
                        train_Dt_end_to_end(
                            downsampling_model = d,
                            upsampling_model = u,
                            model_res = res,
                            flipping = f,
                            multi_step = m
                        )




if __name__ == "__main__":
    component_grid_search_end_to_end()
