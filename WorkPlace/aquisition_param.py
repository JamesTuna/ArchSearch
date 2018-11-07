import numpy as np

from NASsearch.acquisition_func import AcquisitionFunc
from WorkPlace.search import normal_acc
import matplotlib.pyplot as plt


def main():
    train_X, train_y, train_mean, train_std = normal_acc(["./cache/stage_1.txt"])
    current_optimal = np.max(train_y)
    acquisition = AcquisitionFunc(train_X, train_y, current_optimal, mode="ucb", trade_off=0.01)

    test_archs = []
    test_y = []
    with open("b2.txt", "r") as f:
        for line in f.readlines():
            test_archs.append(line.split(" accuracy: ")[0])
            test_y.append(float(line.split(" accuracy: ")[1]))

    # with open("block4_sample.txt", "r") as f:
    #     for line in f.readlines():
    #         test_archs.append(line.split(" accuracy: ")[0])
    #         test_y.append(float(line.split(" accuracy: ")[1]))

    acquisition_value, mean, std = acquisition.compute(test_archs, weight_file="./cache/weight_1.pkl")
    print(std)
    plt.scatter(acquisition_value[:], test_y[:], marker='.')
    # plt.scatter(acquisition_value[256:], test_y[256:], marker='.', color="r")
    plt.show()


if __name__ == "__main__":
    main()