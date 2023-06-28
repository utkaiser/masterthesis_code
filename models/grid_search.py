import sys

from models.baseline_old_paper import train_Dt_old_paper
from models.train_end_to_end import train_Dt_end_to_end


def component_grid_search_end_to_end(experiment_index=0):
    """
    experiment_index: (int) number of the experiment, explained in paper

    Returns
    -------
    performs grid search to run end-to-end model on different component configurations
    """

    model_res = 128

    if experiment_index == 0:
        downsampling_models = ["Interpolation"]
        upsampling_models = ["UNet3"]

    elif experiment_index == 1:
        downsampling_models = ["Interpolation"]
        upsampling_models = [
            "UNet3",
            "UNet6",
            "Tiramisu",
            "UTransform",
        ]
    else:  # experiment 2 -- 5
        downsampling_models = ["Interpolation", "CNN"]
        upsampling_models = ["UNet3"]

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
            train_Dt_end_to_end(
                downsampling_model=d,
                upsampling_model=u,
                model_res=model_res,
                flipping=flipping,
                multi_step=multi_step,
                experiment_index=experiment_index,
                weighted_loss=weighted_loss,
                logging_bool=False,
                visualize_res_bool=False,
                vis_save=False,
            )

    # train_Dt_old_paper(
    #     flipping=flipping,
    #     experiment_index=experiment_index,
    #     visualize_res_bool=True,
    #     vis_save=False
    # )


if __name__ == "__main__":
    # experiment_index = int(sys.argv[1])

    component_grid_search_end_to_end(experiment_index=0)  # experiment_index
