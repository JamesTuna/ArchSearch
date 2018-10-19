import json

import numpy as np

from dependency.label_compressor import LabelCompressor
from dependency.network import Network


class all_Kernel:
    def __init__(self, N, netlist, level):

        net_list = []

        for net in netlist:
            if isinstance(net, str):
                net = json.loads(net)
            net_list.append(net)

        self.N = N
        label_compressor = LabelCompressor()

        self.net = []

        for net in net_list:
            self.net.append(Network(net,level,label_compressor,len(net["normal"])//2))




    def run(self):

        label_set = set()
        net_vector = []

        for net in self.net:
            label_set = list(set(label_set) | set(net.node_label))
        for net in self.net:
            net_vector.append(net.cal_graph_vector(label_set))
 
        for i in range(self.N):
            for net in self.net:
                net.run_iteration()
            label_set = set()
            for net in self.net:
                label_set = list(set(label_set) | set(net.node_label))

            for i in range(len(self.net)):
                net_vector[i] += self.net[i].cal_graph_vector(label_set)


        net_vector = np.array(net_vector)
        return net_vector