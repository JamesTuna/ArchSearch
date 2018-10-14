import json

import numpy as np

from NASsearch.label_compressor import LabelCompressor
from NASsearch.network import Network


class WLKernel:
    def __init__(self, N, net1, net2, level):
        if isinstance(net1, str):
            net1 = json.loads(net1)
        if isinstance(net2, str):
            net2 = json.loads(net2)

        # print("net1", type(net1))
        # print("net2", type(net2))

        self.N = N
        label_compressor = LabelCompressor()
        B1 = len(net1["normal"])//2
        B2 = len(net2["normal"])//2
        self.net1 = Network(net1, level, label_compressor, B1)
        self.net2 = Network(net2, level, label_compressor, B2)

    def run(self):
        label_set = list(set(self.net1.node_label) | set(self.net2.node_label))
        net1_vector = self.net1.cal_graph_vector(label_set)
        net2_vector = self.net2.cal_graph_vector(label_set)
        # print(net1_vector)
        # print(net2_vector)
        # print(self.net1.node_label)
        # print(self.net2.node_label)
        for i in range(self.N):
            self.net1.run_iteration()
            self.net2.run_iteration()
            # print(self.net1.node_label)
            # print(self.net2.node_label)
            label_set = list(set(self.net1.node_label) | set(self.net2.node_label))
            # print(label_set)
            net1_vector += self.net1.cal_graph_vector(label_set)
            net2_vector += self.net2.cal_graph_vector(label_set)
            # print(net1_vector)
            # print(net2_vector)

        net1_vector = np.array(net1_vector)
        net2_vector = np.array(net2_vector)
        # print(net1_vector)
        # print(net2_vector)

        # return np.dot(net1_vector, net2_vector)
        return np.sqrt(np.sum((net1_vector-net2_vector)**2))