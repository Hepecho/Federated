from torchvision import datasets, transforms
import collections

import matplotlib.pyplot as plt
import numpy as np


def cifar10_sample(dataset, num_works):
    # num_items = int(len(dataset) / num_works)
    num_items = 1000
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_works):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_dataset(config):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
    test_dataset = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)

    dict_users = cifar10_sample(train_dataset, config.workers)
    if config.attack == 'raw':
        client_type_list = [0] * config.workers
    else:
        correct_workers = config.workers - config.byzantine_workers
        client_type_list = [0] * correct_workers + [1] * config.byzantine_workers
    return train_dataset, test_dataset, dict_users, client_type_list


if __name__ == '__main__':
    import Config
    xconfig = Config.Config()
    get_dataset(xconfig)

