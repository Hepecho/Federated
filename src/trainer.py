import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import pickle
import copy

from runx.logx import logx
from os.path import join as ospj
from utils import *

from load_dataset import get_dataset
from utils import set_config_name

import Model
from Update import LocalUpdate
from Aggregation import FedAvg, KrumAvg


def evaluate(server_model, datatest, config):
    server_model.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=config.batch_size)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if config.device != 'cpu':
            data, target = data.cuda(), target.cuda()
        log_probs = server_model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    # if config.verbose:
    #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #         test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy.numpy(), test_loss


def train_model(config):
    # 初始化参数服务器
    if config.model_name == 'CNN':
        server_model = Model.CNN(config)
        server_model.to(config.device)
    else:
        exit('Error: Unknown Model Name!')

    # 获取数据集
    train_dataset, test_dataset, dict_users, client_type_list = get_dataset(config)

    config_name = set_config_name(config)  # 关键超参数配置
    local_ckpt_dir = ospj(config.ckpt_dir, config.attack, config_name)
    local_log_dir = ospj(config.log_dir, config.attack, config_name)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    os.makedirs(local_log_dir, exist_ok=True)
    best_model_path = ospj(local_ckpt_dir, 'best_model.pkl')

    # last_model_path = ospj(config.ckpt_dir, config.attack, 'last_' + model_name)
    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Start Train Model [{}]======================'.format(localtime))
    server_model.train()

    train_cache = {'acc': [], 'loss': []}
    test_cache = {'acc': [], 'loss': []}
    best_test_loss = float('inf')

    # copy weights
    clients_global = server_model.state_dict()

    # training

    w_locals = [clients_global for i in range(config.workers)]
    for epoch in range(config.server_epochs):
        start_time = time.time()
        loss_locals = []
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in range(config.workers):
            local = LocalUpdate(
                config=config, dataset=train_dataset, idxs=dict_users[idx], client_type=client_type_list[idx])
            weight, loss = local.train(net=copy.deepcopy(server_model).to(config.device))
            w_locals[idx] = copy.deepcopy(weight)
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        server_lr = config.server_lr * (config.server_rate / (config.server_rate + epoch * config.client_epochs))
        if config.defence == 'raw':
            clients_global = FedAvg(w_locals, server_model.state_dict(), server_lr)
        else:
            clients_global = KrumAvg(
                w_locals, server_model.state_dict(), server_lr, config.workers - config.byzantine_workers - 2, config.k)

        # copy weight to server_model
        server_model.load_state_dict(clients_global)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if epoch > 0 and epoch % 100 == 0:
            epoch_model_path = ospj(local_ckpt_dir, 'iter_model_' + str(epoch) + '.pkl')
            with open(epoch_model_path, 'wb') as f:
                pickle.dump(server_model, f)

        if epoch % config.test_frequency == 0:
            server_model.eval()
            acc_train, loss_train = evaluate(server_model, train_dataset, config)
            acc_test, loss_test = evaluate(server_model, test_dataset, config)
            train_cache['loss'].append(loss_train)
            train_cache['acc'].append(acc_train)
            logx.msg('Epoch: {} | Epoch Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
            logx.msg('Train Loss: {} | Train Acc: {}%'.format(loss_train, acc_train * 100))

            test_cache['loss'].append(loss_test)
            test_cache['acc'].append(acc_test)
            logx.msg('Test Loss: {} | Test Acc: {}%'.format(loss_test, acc_test * 100))
            if loss_test < best_test_loss:
                with open(best_model_path, 'wb') as f:
                    pickle.dump(server_model, f)
                best_test_loss = loss_test

    save_csv(train_cache, ospj(local_log_dir, 'train_' + str(config.load_ckpt) + '.csv'))
    save_csv(test_cache, ospj(local_log_dir, 'test_' + str(config.load_ckpt) + '.csv'))
