import argparse
import json
import random

import subprocess

import os

import copy

import numpy as np

from NASsearch.acquisition_func import AcquisitionFunc
from NASsearch.wl_kernel import WLKernel
from WorkPlace import train_eval

OPERATIONS = ['avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7',
              'dil_conv_3x3', 'conv_7x1_1x7']


def generate_combination(arch, b):
    archs = []
    for op1 in OPERATIONS:
        for op2 in OPERATIONS:
            for op1_conn in range(b + 1):
                for op2_conn in range(b + 1):
                    temp = copy.deepcopy(arch)
                    temp["normal"].append((op1, op1_conn))
                    temp["normal"].append((op2, op2_conn))
                    temp["reduce"].append((op1, op1_conn))
                    temp["reduce"].append((op2, op2_conn))
                    temp["normal_concat"] = list(range(2, b + 2))
                    temp["reduce_concat"] = list(range(2, b + 2))
                    archs.append(json.dumps(temp))
    return archs


def read_block_file(file_name):
    train_X = []
    train_y = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            train_X.append(line.split(" accuracy: ")[0])
            train_y.append(float(line.split(" accuracy: ")[1]))

    return train_X, train_y


def run_gp():
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str, help='structure indicators')
    parser.add_argument('read_file', type=str, help='structure indicators')
    parser.add_argument('out_file', type=str, help='structure indicators')
    args = parser.parse_args()

    train_X, train_y = read_block_file(args.read_file)
    current_optimal = np.max(train_y)
    acquisition = AcquisitionFunc(train_X, train_y, current_optimal, mode="pi", trade_off=0.01)

    arch = json.loads(args.arch)
    archs = generate_combination(arch, 2)
    eliminate_duplicate(archs)
    for a in archs:
        acquisition_value = acquisition.compute(a)
        print(acquisition_value)

        with open(args.out_file, "a") as f:
            f.write(a + " acquisition: " + str(acquisition_value) + "\n")


def random_select(size):
    total_size = 136 * 300
    sampled_archs_index = []
    for i in range(size):
        sampled_archs_index.append(random.randint(0, total_size - 1))

    counter = 0
    for i in range(9):
        with open("block2_arch_" + str(i) + ".txt", "r") as f:
            for line in f.readlines():
                if counter in sampled_archs_index:
                    with open("block2_sample.txt", "a") as f2:
                        f2.write(line)
                counter += 1


def sort_arch():
    my_list = List(256)

    for i in range(9):
        with open("block2_arch_" + str(i) + ".txt", "r") as f:
            for line in f.readlines():
                arch = line.split(" acquisition: ")[0]
                value = float(line.split(" acquisition: ")[1])

                my_list.insert(dict(
                    arch=arch,
                    value=value,
                ))

    with open("block2_arch.txt", "w") as f:
        for item in my_list.array:
            f.write(item["arch"] + " acquisition: " + str(item["value"]) + "\n")


# def eliminate_duplicate():
#     archs = []
#     with open("block1.txt", "r") as f:
#         for line in f.readlines():
#             arch = line.split(" accuracy: ")[0]
#             acc = float(line.split(" accuracy: ")[1])
#             archs.append(dict(
#                 arch=arch,
#                 acc=acc,
#             ))
#
#     indexes = []
#     for i in range(len(archs)):
#         for j in range(i+1, len(archs)):
#             kernel = WLKernel(3, archs[i]["arch"], archs[j]["arch"], 2)
#             dist = kernel.run()
#             if dist == 0 and j not in indexes:
#                 indexes.append(j)
#     indexes.sort()
#     for i in reversed(indexes):
#         archs.pop(i)
#
#     print(len(archs))
#     with open("block1_unique.txt", "a") as f:
#         for arch in archs:
#             f.write(arch["arch"]+" accuracy: "+str(arch["acc"])+"\n")

def eliminate_duplicate(archs):
    indexes = []
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            kernel = WLKernel(3, archs[i], archs[j], 3)
            dist = kernel.run()
            if dist == 0 and j not in indexes:
                indexes.append(j)
    indexes.sort()
    for i in reversed(indexes):
        archs.pop(i)


class List:
    def __init__(self, size=256):
        self.size = size
        self.array = []

    def insert(self, x):
        flag = 0
        for i in range(len(self.array)):
            if x["value"] > self.array[i]["value"]:
                flag = 1
                self.array.insert(i, x)
                break
        if flag == 0:
            self.array.append(x)

        if len(self.array) > self.size:
            self.array = self.array[:self.size]


# def test():
#     my_list = List()
#     for i in range(500):
#         rand = random.randint(1, 100)
#         my_list.insert(rand)
#     print(my_list.array)
#     print(len(my_list.array))


def progressive_search():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="block2_arch.txt", type=str,
                        help='the file contains the architectures for networks to train')
    parser.add_argument('--start', type=int, help='0-256')
    parser.add_argument('--end', type=int, help='0-256')
    parser.add_argument('--index', type=int, help='which gpu to run on')
    parser.add_argument('--outfile', type=str, help='the output file name to store the accuracy')
    args = parser.parse_args()
    # arch = dict(
    #     normal=[],
    #     normal_concat=[],
    #     reduce=[],
    #     reduce_concat=[],
    # )
    #
    # archs = generate_combination(arch, 1)
    # for a in archs:
    #     accr = train_eval.main(a, 3).item()
    #     with open('block1.txt', 'a') as f:
    #         f.write(a + " accuracy: " + str(accr) + "\n")

    archs = []
    with open(args.filename, "r") as f:
        for line in f.readlines():
            arch = line.split(" acquisition: ")[0]
            archs.append(arch)

    for a in archs[args.start:args.end]:
        accr = train_eval.main(a, args.index).item()
        with open(args.outfile, 'a') as f:
            f.write(a + " accuracy: " + str(accr) + "\n")


if __name__ == "__main__":
    progressive_search()
    # run_gp()
    # eliminate_duplicate()
    # sort_arch()
    # random_select(256)
