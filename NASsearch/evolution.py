import json
import os

import numpy as np
from torch.multiprocessing import Process
import subprocess

from NASsearch import train_eval
from NASsearch.acquisition_func import AcquisitionFunc

NASNet = dict(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = dict(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

PNASNet = dict(
    normal=[
        ('sep_conv_7x7', 1),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 0),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_7x7', 1),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('max_pool_3x3', 1),
        ('skip_connect', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 0),
    ],
    reduce_concat=[2, 3, 4, 5, 6],
)


def init_pool(N, operations, B):
    op_num = 2 * B
    pop = []
    for n in range(N):
        normal = []
        reduce = []
        normal_concat = list(range(2, B + 2))
        reduce_concat = list(range(2, B + 2))
        for i in range(op_num):
            normal_rand = np.random.randint(0, len(operations) - 1)
            reduce_rand = np.random.randint(0, len(operations) - 1)
            normal_op = np.random.randint(0, i // 2 + 1)
            reduce_op = np.random.randint(0, i // 2 + 1)
            normal.append((operations[normal_rand], normal_op))
            reduce.append((operations[reduce_rand], reduce_op))

            try:
                normal_concat.remove(normal_op)
                reduce_concat.remove(reduce_op)
            except ValueError:
                pass
        net = dict(
            normal=normal,
            normal_concat=normal_concat,
            reduce=reduce,
            reduce_concat=reduce_concat,
        )
        net = json.dumps(net)
        individual = dict(
            net=net,
            acquisition=None,
        )
        pop.append(individual)
    return pop


def sample(pop, S):
    samples = []
    for s in range(S):
        i1 = np.random.randint(0, len(pop) - 1)
        i2 = np.random.randint(0, len(pop) - 1)
        if pop[i1]["acquisition"] > pop[i2]["acquisition"]:
            samples.append(pop[i1])
        else:
            samples.append(pop[i2])
    return samples


def change_op(individual, operations):
    # print(type(individual))
    individual = json.loads(individual)

    op = np.random.randint(0, len(individual["normal"]) * 2 - 1)
    op_rand = np.random.randint(0, len(operations) - 1)
    if op // len(individual["normal"]) == 0:
        individual["normal"][op % len(individual["normal"])] = (
            operations[op_rand],
            individual["normal"][op % len(individual["normal"])][1]
        )
    else:
        individual["reduce"][op % len(individual["normal"])] = (
            operations[op_rand],
            individual["reduce"][op % len(individual["normal"])][1]
        )
    return json.dumps(individual)


def change_connection(individual, B):
    # print(type(individual))
    individual = json.loads(individual)
    op = np.random.randint(0, len(individual["normal"]) * 2 - 1)
    i = op % len(individual["normal"])
    rand = np.random.randint(0, i // 2 + 1)
    if op // len(individual["normal"]) == 0:
        individual["normal"][op % len(individual["normal"])] = (
            individual["normal"][op % len(individual["normal"])][0],
            rand
        )
        individual["normal_concat"] = change_concat(B, individual["normal"])
    else:
        individual["reduce"][op % len(individual["normal"])] = (
            individual["reduce"][op % len(individual["normal"])][0],
            rand
        )
        individual["reduce_concat"] = change_concat(B, individual["reduce"])
    return json.dumps(individual)


def change_concat(B, cell):
    concat = list(range(2, B + 2))
    for c in cell:
        try:
            concat.remove(c[1])
        except ValueError:
            pass
    return concat


def mutation(individual, p, operations, B):
    mut_num = np.random.choice(list(range(1, 5)), p=p)
    for i in range(mut_num):
        rand = np.random.rand()
        if rand < 0.5:
            individual = change_op(individual, operations)
        else:
            individual = change_connection(individual, B)
    return individual


def main():
    operations = ['avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7',
                  'dil_conv_3x3', 'conv_7x1_1x7']
    B = 5
    N = 30
    T = 20
    iteration_total = 30
    X_train = []
    y_train = []
    size = 4
    pop = []
    # processes = []
    # for rank in range(size):
    #     p = Process(target=train_eval.dist_main, args=[rank, size, json.dumps(NASNet)])
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # train_eval.dist_main(size, json.dumps(NASNet))

    # workers = 4
    # for rank in range(workers):
    #     subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "NASsearch/train_eval.py"), str(rank), json.dumps(NASNet), str(workers)])
    # start_point = init_pool(
    #     N=1,
    #     operations=operations,
    #     B=B,
    # )
    # print(start_point[0])
    # y_train.append(93.5999984741211)
    # X_train.append(json.dumps(NASNet))
    # with open("train.txt", "a") as f:
    #     f.write(X_train[-1])
    #     f.write(" accuracy: ")
    #     f.write(str(y_train[-1])+"\n")
    # y_train.append(train_eval.dist_main(size, json.dumps(AmoebaNet)).item())
    # X_train.append(json.dumps(AmoebaNet))

    with open("train.txt", "r") as f:
        for line in f.readlines():
            arch = line.split(" accuracy: ")[0]
            acc = line.split(" accuracy: ")[1].split("\n")[0]
            X_train.append(arch)
            y_train.append(float(acc))
    current_optimal = max(y_train)

    # with open("pop.txt", "r") as f:
    #     for line in f.readlines():
    #         net = line.split(" acquisition: ")[0]
    #         acqui = line.split(" acquisition: ")[1].split("\n")[0]
    #         pop.append(dict(
    #             net=net,
    #             acquisition=acqui,
    #         ))

    # with open("train.txt", "a") as f:
    #     f.write(X_train[-1])
    #     f.write(" accuracy: ")
    #     f.write(str(y_train[-1])+"\n")

    pop = init_pool(
        N=N,
        operations=operations,
        B=B,
    )
    for iter in range(iteration_total):
        acquisition = AcquisitionFunc(X_train, y_train, current_optimal, mode="pi", trade_off=0.01)
        for t in range(T):
            for p in pop:
                p["acquisition"] = acquisition.compute(json.loads(p["net"]))
                print(p["acquisition"])

            with open("pop.txt", "w") as f:
                for p in pop:
                    f.write(p["net"])
                    f.write(" acquisition: ")
                    f.write(str(p["acquisition"]) + "\n")

            samples = sample(pop=pop, S=10)
            # print(samples)
            mutations = []
            for s in samples:
                temp = s["net"]
                mutate_net = mutation(
                    individual=temp,
                    p=[0.5, 0.25, 0.125, 0.125],
                    operations=operations,
                    B=B,
                )
                # print(type(mutate_net))
                mutations.append(dict(
                    net=mutate_net,
                    acquisition=None,
                ))
            for m in mutations:
                m["acquisition"] = acquisition.compute(json.loads(m["net"]))
                print(m["acquisition"])

            pop = pop + mutations
            pop = sorted(pop, key=lambda item: item["acquisition"])
            pop = pop[-N:]
        X_train.append(pop[-1]["net"])
        y_train.append(train_eval.dist_main(size, pop[-1]["net"]).item())

        with open("train.txt", "a") as f:
            f.write(X_train[-1])
            f.write(" accuracy: ")
            f.write(str(y_train[-1]) + "\n")


if __name__ == "__main__":
    main()
