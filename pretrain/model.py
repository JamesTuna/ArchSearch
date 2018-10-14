import torch.nn as nn

from operations import FactorizedReduce, ReLUConvBN, OPS
from pretrain.operations import *
from pretrain.utils import drop_path


class Cell(nn.Module):
    def __init__(self, C_op0_prev, C_op1_prev, C, reduction, op0_reduction, op1_reduction, op1_name, op2_name, op0_prev, op1_prev):
        super(Cell, self).__init__()
        self.multiplier = 2
        if reduction:
            stride = 2
        else:
            stride = 1
        self.op0_re = op0_reduction
        self.op1_re = op1_reduction

        if op0_prev == 1 and op1_prev == 2:
            if op0_reduction and not op1_reduction:
                self.preprocess0 = ReLUConvBN(C_op0_prev, C, 1, 1, 0)
                self.preprocess1 = FactorizedReduce(C_op1_prev, C)
            else:
                self.preprocess0 = ReLUConvBN(C_op0_prev, C, 1, 1, 0)
                self.preprocess1 = ReLUConvBN(C_op1_prev, C, 1, 1, 0)
        elif op0_prev == 2 and op1_prev == 1:
            if not op0_reduction and op1_reduction:
                self.preprocess0 = FactorizedReduce(C_op0_prev, C)
                self.preprocess1 = ReLUConvBN(C_op1_prev, C, 1, 1, 0)
            else:
                self.preprocess0 = ReLUConvBN(C_op0_prev, C, 1, 1, 0)
                self.preprocess1 = ReLUConvBN(C_op1_prev, C, 1, 1, 0)
        else:
            self.preprocess0 = ReLUConvBN(C_op0_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConvBN(C_op1_prev, C, 1, 1, 0)

        # if op0_reduction and op1_reduction:
        #     self.preprocess0 = ReLUConvBN(op0_prev, C, 1, 1, 0)
        #     self.preprocess1 = ReLUConvBN(op1_prev, C, 1, 1, 0)
        # if op0_reduction and not op1_reduction:
        #     self.preprocess0 = ReLUConvBN(op0_prev, C, 1, 1, 0)
        #     self.preprocess1 = FactorizedReduce(op1_prev, C)
        # elif not op0_reduction and op1_reduction:
        #     self.preprocess0 = FactorizedReduce(op0_prev, C)
        #     self.preprocess1 = ReLUConvBN(op1_prev, C, 1, 1, 0)
        # else:
        #     self.preprocess0 = ReLUConvBN(op0_prev, C, 1, 1, 0)
        #     self.preprocess1 = ReLUConvBN(op1_prev, C, 1, 1, 0)
        self.op1 = OPS[op1_name](C, stride, True)
        self.op2 = OPS[op2_name](C, stride, True)

    def forward(self, s0, s1, drop_prob):
        # print(s0.shape, s1.shape)
        # print(self.op0_re, self.op1_re)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # print(s0.shape, s1.shape)

        s0 = self.op1(s0)
        s1 = self.op2(s1)
        if self.training and drop_prob > 0.:
            if not isinstance(self.op1, Identity):
                s0 = drop_path(s0, drop_prob)
            if not isinstance(self.op2, Identity):
                s1 = drop_path(s1, drop_prob)
        # print(s0.shape, s1.shape)

        return s0+s1


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, op1_name, op2_name, op1_prev, op2_prev):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._op1_prev = int(op1_prev)
        self._op2_prev = int(op2_prev)

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        channels = [C_curr]
        reductions = [False]
        C_curr = C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if i+1-self._op1_prev >= 0:
                op1_prev = channels[i+1-self._op1_prev]
                op1_reduction = reductions[i+1-self._op1_prev]
            else:
                op1_prev = channels[0]
                op1_reduction = reductions[0]
            if i+1-self._op2_prev >= 0:
                op2_prev = channels[i+1-self._op2_prev]
                op2_reduction = reductions[i+1-self._op2_prev]
            else:
                op2_prev = channels[0]
                op2_reduction = reductions[0]

            cell = Cell(op1_prev, op2_prev, C_curr, reduction, op1_reduction, op2_reduction, op1_name, op2_name, self._op1_prev, self._op2_prev)
            reductions.append(reduction)
            self.cells += [cell]
            channels.append(C_curr)

        # C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # self.cells = nn.ModuleList()
        # reduction_prev = False
        # for i in range(layers):
        #     if i in [layers // 3, 2 * layers // 3]:
        #         C_curr *= 2
        #         reduction = True
        #     else:
        #         reduction = False
        #     cell = Cell(C_prev_prev, C_prev, C_curr, reduction, reduction_prev, op1_name, op2_name)
        #     reduction_prev = reduction
        #     self.cells += [cell]
        #     C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        #     if i == 2 * layers // 3:
        #         C_to_auxiliary = C_prev

        # if auxiliary:
        #     self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, input):
        logits_aux = None
        states = [self.stem(input)]
        # print(input)
        # print(s0)
        # print(s1)
        for i, cell in enumerate(self.cells):
            if i+1-self._op1_prev >= 0:
                s0 = states[i+1-self._op1_prev]
            else:
                s0 = states[0]
            if i+1-self._op2_prev >= 0:
                s1 = states[i+1-self._op2_prev]
            else:
                s1 = states[0]
            # print(s0.shape, s1.shape)
            states.append(cell(s0, s1, self.drop_path_prob))
            # print("level: ", i)
            # s0, s1 = s1, cell(s0, s1)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(states[-1])
        out = self.global_pooling(states[-1])
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
