import copy
import torch
from torch import nn
from torch.linalg import vector_norm


def FedAvg(clients, server, server_lr):
    g_avg = copy.deepcopy(clients[0])
    for k in g_avg.keys():
        for i in range(1, len(clients)):
            g_avg[k] += clients[i][k]
        g_avg[k] = torch.div(g_avg[k], len(clients))
        g_avg[k] = server[k] + g_avg[k] * server_lr
    return g_avg


def KrumAvg(clients, server, server_lr, max_num, topk=1):
    num_workers = len(clients)
    g_avg = copy.deepcopy(clients[0])
    scores = []
    # print(g_avg.keys())
    dist_ij = torch.zeros(num_workers, num_workers).to(g_avg['conv1.0.weight'].device)
    for i in range(num_workers):
        for j in range(num_workers):
            if j == i:
                continue
            if j < i:
                dist_ij[i][j] = dist_ij[j][i]
            for k in g_avg.keys():
                dist_ij[i][j] += vector_norm(clients[i][k] - clients[j][k], ord=2)
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

