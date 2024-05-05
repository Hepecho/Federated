#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, label_flipping=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.label_flipping = label_flipping

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.label_flipping:
            label = 9 - label
        return image, label


class LocalUpdate(object):
    def __init__(self, config, dataset=None, idxs=None, client_type=0):
        self.config = config
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.client_type = client_type
        if self.config.attack == 'lf' and self.client_type == 1:
            label_flipping = True
        else:
            label_flipping = False
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, label_flipping), batch_size=self.config.batch_size, shuffle=True)

    def train(self, net):
        delta_net = copy.deepcopy(net.state_dict())
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.config.client_lr, weight_decay=1e-4)

        epoch_loss = []
        for iter in range(self.config.client_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.config.device), labels.to(self.config.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.config.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        new_net = net.state_dict()
        if self.client_type == 1:
            for k in delta_net.keys():
                delta_net[k] = (new_net[k] - delta_net[k]) * self.config.attack_strength
        else:
            for k in delta_net.keys():
                delta_net[k] = new_net[k] - delta_net[k]

        return delta_net, sum(epoch_loss) / len(epoch_loss)
