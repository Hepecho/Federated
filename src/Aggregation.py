import copy
import torch
from torch import nn
from torch.linalg import vector_norm


def FedAvg(clients, server, server_lr):
    """
    :param clients: 工作节点回传的梯度集合，有序字典构成的list
    :param server: 参数服务器的当前参数，有序字典
    :param server_lr: 当前server的学习率
    :return: 平均聚合后的参数，有序字典
    """
    g_avg = copy.deepcopy(clients[0])
    for k in g_avg.keys():
        for i in range(1, len(clients)):
            g_avg[k] += clients[i][k]
        g_avg[k] = torch.div(g_avg[k], len(clients))
        g_avg[k] = server[k] + g_avg[k] * server_lr
    return g_avg


def KrumAvg(clients, server, server_lr, max_num, topk=1):
    """
    :param clients: 工作节点回传的梯度集合，有序字典构成的list
    :param server: 参数服务器的当前参数，有序字典
    :param server_lr: 当前server的学习率
    :param max_num: 由n-b-2得到，是krum考虑的最接近梯度向量的数量
    :param topk: muti-krum的插值参数，当topk=1时即为krum，topk=n时即为FedAvg
    :return: Krum聚合后的参数，有序字典
    """
    num_workers = len(clients)
    g_avg = copy.deepcopy(clients[0])
    scores = []
    dist_ij = torch.zeros(num_workers, num_workers).to(g_avg['conv1.0.weight'].device)
    for i in range(num_workers):
        for j in range(i + 1, num_workers):  # dist_ij[i][j] == dist[j][i]
            for k in g_avg.keys():
                dist_ij[i][j] += vector_norm(clients[i][k] - clients[j][k], ord=2)
            dist_ij[j][i] = dist_ij[i][j]
        dist_ij[i].sort()
        scores.append(sum(dist_ij[i][1:max_num + 1]))  # 忽略i==j

    sorted_id = sorted(range(len(scores)), key=lambda x: scores[x])

    g_avg = copy.deepcopy(clients[sorted_id[0]])
    for k in g_avg.keys():
        for i in range(1, topk):
            g_avg[k] += clients[sorted_id[i]][k]
        g_avg[k] = torch.div(g_avg[k], topk)
        g_avg[k] = server[k] + g_avg[k] * server_lr

    return g_avg

