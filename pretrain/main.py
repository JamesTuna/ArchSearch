import json
import subprocess

import os


def run():
    trained_net = []
    with open("accuracy_3.txt", "r") as f:
        for line in f.readlines():
            network = line.split(" tensor")[0]
            trained_net.append(network)

    net_to_train = []
    with open("networks.txt", "r") as f:
        for line in f.readlines():
            net = line.split("\n")[0]
            if net not in trained_net:
                net_to_train.append(net)

    # print(len(net_to_train))
    lenth = len(net_to_train)
    # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[:lenth//4]), str(2)])
    subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[lenth//4:2*lenth//4]), str(5)])
    subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[2*lenth//4:3*lenth//4]), str(6)])
    subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[3*lenth//4:]), str(7)])
    # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[4*lenth//8:5*lenth//8]), str(4)])
    # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[5*lenth//8:6*lenth//8]), str(5)])
    # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[6*lenth//8:7*lenth//8]), str(6)])
    # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(net_to_train[7*lenth//8:]), str(7)])
    # operations = ["sepconv3", "sepconv5", "sepconv7", "dilconv3", "dilconv5", "maxpooling", "avgpooling", "identity"]
    # counter = 0
    # networks = []
    # for op1 in operations:
    #     for op2 in operations:
    #         for op1_prev in range(2):
    #             for op2_prev in range(2):
    #                 counter += 1
    #                 networks.append(dict(
    #                     op1=op1,
    #                     op2=op2,
    #                     op1_prev=op1_prev+1,
    #                     op2_prev=op2_prev+1,
    #                 ))
    #                 with open("networks.txt", "a") as f:
    #                     f.write(json.dumps(dict(
    #                         op1=op1,
    #                         op2=op2,
    #                         op1_prev=op1_prev + 1,
    #                         op2_prev=op2_prev + 1,
    #                     ))+"\n")
    #                 if counter % 128 == 0:
    #                     if counter//32%4 == 0:
    #                         print(networks[0])
                        # subprocess.Popen(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps(networks), str(counter//128-1)])
                        # networks = []
    # subprocess.run(["python3.6", os.path.join(os.getcwd(), "pretrain/workers.py"), json.dumps([dict(
    #     op1="dilconv3",
    #     op2="sepconv3",
    #     op1_prev="2",
    #     op2_prev="1",
    # )]), str(1)])


if __name__ == "__main__":
    run()
