from os.path import join as ospj
import torch


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'CNN'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log_dir = ospj('log', self.model_name)
        self.ckpt_dir = ospj('checkpoint', self.model_name)
        self.attack = 'bf'  # 攻击模式 ['raw', 'lf', 'bf']
        self.defence = 'krum'  # 防御算法 ['raw, 'krum']

        # 学习参数
        self.client_epochs = 10  # client epoch数  25
        self.server_epochs = 100  # server 聚合更新次数
        self.batch_size = 100  # mini-batch大小  128
        self.client_lr = 0.15  # 学习率 alpha  0.01
        self.server_lr = 0.15  # server 学习率 0.1
        self.server_rate = 2000  # server 学习率递减参数
        self.load_ckpt = 0  # 从第几个epoch处继续训练

        # 网络参数
        self.workers = 20  # 工作节点数量 n [10, 100]
        self.byzantine_workers = 8  # 拜占庭节点数量 f
        if self.attack == 'raw':
            self.attack_strength = 1.0
        elif self.attack == 'lf':  # 拜占庭节点的梯度缩放幅度 绝对值取值[0.01, 0.1, 1.0, 5.0, 10]
            self.attack_strength = 1.0
        else:
            self.attack_strength = -1.0

        # defence参数
        self.k = 1  # muti-krum的参数，当k=1时，等于krum，当k=n时，等价于FedAvg
        self.b = 4  # krum假定的拜占庭节点数量，要求2b + 2 < n，这里n==20时，b最多取8。根据krum的原理，b越小，效果越好

        # 输出信息
        self.test_frequency = 1  # test频率
