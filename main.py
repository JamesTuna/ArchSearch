from GradientForGP.gp_grad import gp_gradient
from Kernel_optimization.all_gp import save_vector_label
from reweightForWL.gp_wl import reweight


def run():
    # save vectors and labels for stage i
    # save_vector_label(
    #     weight_file=None,
    #     stage_file="./cache/stage_2.txt",
    #     label_file="./cache/label_2.pkl",
    #     dist_mat_file="./cache/dist_mat_2.pkl",
    #     vector_file="./cache/vector_2.pkl"
    # )

    # use soml to optimize weights
    # reweight(
    #     load_weight=False,
    #     weight_file="./cache/weight_2.pkl",
    #     vector_file="./cache/vector_2.pkl",
    #     label_file="./cache/label_2.pkl",
    # )

    # recalculate dist mat calculated using optimised weights
    # save_vector_label(
    #     weight_file="./cache/weight_2.pkl",
    #     stage_file="./cache/stage_2.txt",
    #     label_file="./cache/label_2.pkl",
    #     dist_mat_file="./cache/dist_mat_2.pkl",
    #     vector_file="./cache/vector_2.pkl"
    #
    # )

    # optimize hyper-parameters of gp process
    gp_gradient(
        dist_mat_file="./cache/dist_mat_2.pkl",
        stage_file="./cache/stage_2.txt",
    )
    # run gp prdiction
    # sort archs
    # eliminate duplicates
    # train archs


if __name__ == "__main__":
    run()