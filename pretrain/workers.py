import argparse
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable

from pretrain import utils
from pretrain.model import Network

AUXILIARY = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('networks', help='structure indicators')
    parser.add_argument('index', help='index')
    args = parser.parse_args()
    # print(json.loads(args.networks))

    for network in json.loads(args.networks):
        net = json.loads(network)
        # net = network
        accuracy = train_eval(net["op1"], net["op2"], net["op1_prev"], net["op2_prev"], int(args.index))

        with open("accuracy_3.txt", "a") as myfile:
            myfile.write(json.dumps(net) + " " + str(accuracy) + "\n")
            # myfile.write(json.dumps(network)+"\n")


def train_eval(op1_name, op2_name, op1_prev, op2_prev, index):
    torch.cuda.set_device(index)
    #
    cudnn.benchmark = True
    cudnn.enabled = True

    model = Network(
        C=36,
        num_classes=10,
        layers=20,
        auxiliary=False,
        op1_name=op1_name,
        op2_name=op2_name,
        op1_prev=op1_prev,
        op2_prev=op2_prev,
    )
    # print(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4,
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(cutout=False, cutout_length=16)
    train_data = dset.CIFAR10(root="data/", train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root="data/", train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
    # train_data, batch_size=16, shuffle=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=2)
    # valid_data, batch_size=16, shuffle=False, num_workers=2)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 36)
    max_valid = 0
    valid_acc = 0

    for epoch in range(36):
        scheduler.step()
        print('epoch {} lr {}'.format(epoch, scheduler.get_lr()[0]))
        print('epoch {}'.format(epoch))
        # model.drop_path_prob = 0.2 * epoch / 36
        model.drop_path_prob = 0

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        print('train_acc {}'.format(train_acc))

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        print('valid_acc {}'.format(valid_acc))

        # if prev_valid_acc > valid_acc:
        #     return prev_valid_acc
        # prev_valid_acc = valid_acc
        if valid_acc > max_valid:
            max_valid = valid_acc

        utils.save(model, 'weights.pt')
    return max_valid


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
        #
        # input = Variable(input)
        # target = Variable(target)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if AUXILIARY:
            loss_aux = criterion(logits_aux, target)
            loss += 0.4 * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % 50 == 0:
            print(loss.data[0])
            print('train {} {} {} {}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)
        #
        # input = Variable(input, volatile=True)
        # target = Variable(target, volatile=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % 50 == 0:
            print('valid {} {} {} {}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
