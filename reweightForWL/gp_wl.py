#! /usr/bin/env python3
import matplotlib.pyplot as plt
import pickle, torch
import numpy as np
from torch.autograd import Variable
import random


def plot(transformed_v, label):
    x = []
    y = []
    for i in range(0, len(label) - 1):
        for j in range(i + 1, len(label)):
            x.append(np.sqrt(np.sum((transformed_v[i] - transformed_v[j]) ** 2)))
            y.append(np.abs(label[i] - label[j]))
    plt.scatter(x, y)
    plt.ion()
    plt.pause(10)
    plt.close()


def batch_opt(batch_v, batch_label, optimizer, un_transformed_v, un_label):
    optimizer.zero_grad()
    loss = torch.Tensor([0])
    pos_th = 0.4
    neg_th = 0.5
    margin = 50
    for i in range(batch_v.shape[0] - 2):
        print(i)
        pos = select_pos(batch_label[i], pos_th, un_label)
        neg = select_neg(batch_label[i], neg_th, un_label)
        if pos == -1 or neg == -1:
            continue
        loss += max(0, torch.sum((batch_v[i] - un_transformed_v[pos]) ** 2) - torch.sum(
            (batch_v[i] - un_transformed_v[neg]) ** 2) - margin)
    print(loss)
    if loss != 0:
        loss.backward()
        optimizer.step()


def select_pos(anchor_y, pos_th, un_label):
    d_y = 10
    unsample_length = un_label.shape[0]
    id_random = random.randint(0, unsample_length - 1)
    count = 0
    while d_y > pos_th:
        id_random = random.randint(0, unsample_length - 1)
        d_y = abs(un_label[id_random] - anchor_y)
        count += 1
        if count >= 300:
            return -1
    return id_random


def select_neg(anchor_y, neg_th, un_label):
    d_y = -10
    unsample_length = un_label.shape[0]
    id_random = random.randint(0, unsample_length - 1)
    count = 0
    while d_y < neg_th:
        id_random = random.randint(0, unsample_length - 1)
        d_y = abs(un_label[id_random] - anchor_y)
        if count >= 300:
            return -1
    return id_random


def reweight(load_weight, weight_file, vector_file, label_file):
    with open(vector_file, 'rb') as f:
        net_v = pickle.load(f)
    with open(label_file, 'rb') as f:
        label = pickle.load(f)

    STEPS = 10000
    LR = 0.01
    N = net_v.shape[0]
    SAMPLES = np.arange(N)

    label = torch.Tensor(label)
    dim = net_v.shape[1]
    weight = Variable(torch.Tensor(np.ones(dim)), requires_grad=True)
    if load_weight:
        with open('./weight.pkl', 'rb') as f:
            weight = pickle.load(f)
            weight = Variable(weight, requires_grad=True)
    net_v = torch.Tensor(net_v).detach()

    optimizer = torch.optim.Adam([weight], lr=LR)

    for i in range(STEPS):
        print(i)
        transformed_v = net_v * weight
        np.random.shuffle(SAMPLES)
        samples = SAMPLES[:10]
        un_samples = SAMPLES[10:]
        batch_opt(transformed_v[samples], label[samples], optimizer, transformed_v[un_samples], label[un_samples])
        print("weight: ")
        print(i, weight)
        if i % 500 == 0:
            plot(transformed_v.data.numpy(), label.data.numpy())
        if i == STEPS - 1:
            with open(weight_file, 'wb') as f:
                pickle.dump(weight, f)

