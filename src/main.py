import torch
import os
from os.path import join as ospj
from runx.logx import logx
import argparse

from trainer import train_model
import Config
from utils import *


def args_parser():
    parser = argparse.ArgumentParser(description='DDL')
    parser.add_argument('--action', type=int, default=0,
                        help="0 means 'raw' attack;")
    # federated arguments
    # parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    # parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    # parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    # parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='CNN', help='model name')
    # parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to use for convolution')
    # parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    # parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    # parser.add_argument('--max_pool', type=str, default='True',
    #                     help="Whether use max pooling rather than strided convolutions")

    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    # parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    # parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args


def main(config, args):
    config_name = set_config_name(config)
    logx.initialize(logdir=ospj(config.log_dir, config.attack, config_name), coolname=False, tensorboard=False)
    logx.msg(str(args))
    logx.msg(str(config.__dict__))

    # 分布式训练
    train_model(config)


if __name__ == '__main__':
    args = args_parser()
    config = Config.Config()

    os.makedirs(ospj(config.log_dir, config.attack), exist_ok=True)
    os.makedirs(ospj(config.ckpt_dir, config.attack), exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    assert args.action in range(16), "args.action must in range(16)!"

    # Basic Mode
    if args.action == 0:
        assert config.attack == 'raw', "config.attack != 'raw'!"
        main(config, args)
        # plt_hp(
        #     title='Federated System',
        #     hp_type='attack',
        #     hp_list=['raw', 'lf', 'bf'],
        #     config=config
        # )
    elif args.action == 1:
        assert config.attack == 'lf', "config.attack != 'lf'!"
        main(config, args)
    elif args.action == 2:
        assert config.attack == 'bf', "config.attack != 'bf'!"
        main(config, args)
    elif args.action == 3:
        assert config.defence == 'krum', "config.defence != 'krum'!"
        main(config, args)

    # Hyperparameter Sensitivity & Plt
    elif args.action == 4:
        assert config.attack == 'lf', "config.attack != 'lf'!"
        for i in range(6, 15, 2):
            config.byzantine_workers = i
            main(config, args)

    elif args.action == 5:
        assert config.attack == 'lf', "config.attack != 'lf'!"
        # 绘制lf在有/无krum防御情况下，有关拜占庭节点数量的学习曲线对比图

        if config.defence == 'raw':
            title = 'Malicious Workers of Label Flipping'
        else:
            title = 'Malicious Workers of Label Flipping with Krum'

        plt_hp(
            title=title,
            hp_type='byzantine_workers',
            hp_list=list(range(6, 15, 2)),
            config=config
        )

    elif args.action == 6:
        assert config.attack == 'lf', "config.attack != 'lf'!"
        for p in [0.01, 0.1, 1.0, 5.0, 10]:
            config.attack_strength = p
            main(config, args)

    elif args.action == 7:
        assert config.attack == 'lf', "config.attack != 'lf'!"
        # 绘制lf在有/无krum防御情况下，有关攻击强度的学习曲线对比图

        if config.defence == 'raw':
            title = 'Attack Strength of Label Flipping'
        else:
            title = 'Attack Strength of Label Flipping with Krum'

        plt_hp(
            title=title,
            hp_type='attack_strength',
            hp_list=[0.01, 0.1, 1.0, 5.0, 10],
            config=config
        )

    elif args.action == 8:
        assert config.attack == 'bf', "config.attack != 'bf'!"
        for i in range(6, 15, 2):
            config.byzantine_workers = i
            main(config, args)
        # config.byzantine_workers = 8
        # for p in [0.01, 0.1, 5.0, 10.0]:
        #     config.attack_strength = - p
        #     main(config, args)

    elif args.action == 9:
        assert config.attack == 'bf', "config.attack != 'bf'!"
        # 绘制bf在有/无krum防御情况下，有关拜占庭节点数量的学习曲线对比图

        if config.defence == 'raw':
            title = 'Malicious Workers of Bit Flipping'
        else:
            title = 'Malicious Workers of Bit Flipping with Krum'

        plt_hp(
            title=title,
            hp_type='byzantine_workers',
            hp_list=list(range(6, 15, 2)),
            config=config
        )

    elif args.action == 10:
        assert config.attack == 'bf', "config.attack != 'bf'!"
        for p in [0.01, 0.1, 5.0, 10.0]:
            config.attack_strength = - p
            main(config, args)

    elif args.action == 11:
        assert config.attack == 'bf', "config.attack != 'bf'!"
        # 绘制bf在有/无krum防御情况下，有关攻击强度的学习曲线对比图

        if config.defence == 'raw':
            title = 'Attack Strength of Bit Flipping'
        else:
            title = 'Attack Strength of Bit Flipping with Krum'

        plt_hp(
            title=title,
            hp_type='attack_strength',
            hp_list=[-0.01, -0.1, -1.0, -5.0, -10.0],
            config=config
        )

    elif args.action == 12:
        assert config.defence == 'krum', "config.defence != 'krum'!"
        for k in [5, 10, 15, 20]:
            config.k = k
            main(config, args)

    elif args.action == 13:
        # 绘制krum的参数k的学习曲线对比图
        assert config.defence == 'krum', "config.defence != 'krum'"
        if config.attack == 'raw':
            title = 'Krum Parameter k'
        elif config.attack == 'lf':
            title = 'Krum Parameter k under Label Flipping'
        else:
            title = 'Krum Parameter k under Bit Flipping'

        plt_hp(
            title=title,
            hp_type='k',
            hp_list=[1, 5, 10, 15, 20],
            config=config
        )

    elif args.action == 14:
        assert config.defence == 'krum', "config.defence != 'krum'!"
        for b in range(6, 13, 2):
            config.b = b
            main(config, args)

    elif args.action == 15:
        # 绘制krum的参数b的学习曲线对比图
        assert config.defence == 'krum', "config.defence != 'krum'"
        if config.attack == 'raw':
            title = 'Krum Parameter b'
        elif config.attack == 'lf':
            title = 'Krum Parameter b under Label Flipping'
        else:
            title = 'Krum Parameter b under Bit Flipping'

        plt_hp(
            title=title,
            hp_type='b',
            hp_list=list(range(4, 13, 2)),
            config=config
        )

    else:
        pass

