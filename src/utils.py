import pandas as pd

import numpy as np
from os.path import join as ospj
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_csv(cache, csv_path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(csv_path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk', index=False)
    f1.close()


def read_csv(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer')
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data


def save_json(cache, json_path):
    # 保存文件
    tf = open(json_path, "w")
    tf.write(json.dumps(cache))
    tf.close()


def read_json(json_path):
    # 读取文件
    tf = open(json_path, "r")
    new_dict = json.load(tf)
    return new_dict


def plt_line_chart(metric_data, img_path):
    color_par = ['#D62728', '#1F77B4', '#FF7F0E', '#2CA02C', '#8A2AA0', '#7E84F7', '#F248BF']

    marker_par = ['.', 'o', 'v', 's', 'p', '*']

    idx_p = 0
    for i, k in enumerate(metric_data.keys()):
        if k not in ['x', 'xlabel', 'ylabel', 'title']:
            plt.plot(
                metric_data['x'], metric_data[k],
                color=color_par[idx_p],
                alpha=1, linewidth=1, label=k
            )
            idx_p = (idx_p + 1) % len(color_par)

    plt.legend()  # 显示图例
    plt.grid(ls='--')  # 生成网格
    plt.xlabel(metric_data['xlabel'])
    plt.ylabel(metric_data['ylabel'])
    plt.title(metric_data['title'])
    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    if metric_data['ylabel'] == 'Accuracy':
        y_major_locator = MultipleLocator(0.1)
        # 把y轴的刻度间隔设置为0.1，并存在变量里
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为x_major_locator的倍数
        ax.yaxis.set_major_locator(y_major_locator)
        # 把y轴的主刻度设置为y_major_locator的倍数
        plt.ylim(0.05, 0.70)
    plt.savefig(img_path)
    plt.clf()


def set_config_name(config):
    if config.defence == 'raw':
        name = f'bw{config.byzantine_workers}_as{config.attack_strength}'
    else:
        name = f'bw{config.byzantine_workers}_as{config.attack_strength}_k{config.k}_b{config.b}'
    return name


def plt_hp(title, hp_type, hp_list, config):
    """"""
    # acc
    metric_data = {
        'x': [n for n in range(100)],
        'xlabel': 'Epoch',
        'ylabel': 'Accuracy',
        'title': title
    }
    for i in hp_list:
        config.__dict__[hp_type] = i
        if i == 'bf':
            config.attack_strength = - 1.0
        config_name = set_config_name(config)
        metric_file_path = ospj(config.log_dir, config.attack, config_name, 'test_0.csv')
        if i == 'bf':
            config.attack_strength = 1.0
        test_pd = read_csv(metric_file_path)
        metric_data[f'{hp_type} = {i}'] = test_pd['acc']
    plt_line_chart(metric_data, img_path=f'image/{title} (Acc).png')

    # loss
    metric_data = {
        'x': [n for n in range(100)],
        'xlabel': 'Epoch',
        'ylabel': 'Loss',
        'title': title
    }
    for i in hp_list:
        config.__dict__[hp_type] = i
        if i == 'bf':
            config.attack_strength = - 1.0
        config_name = set_config_name(config)
        metric_file_path = ospj(config.log_dir, config.attack, config_name, 'test_0.csv')
        if i == 'bf':
            config.attack_strength = 1.0
        test_pd = read_csv(metric_file_path)
        metric_data[f'{hp_type} = {i}'] = test_pd['loss']
    plt_line_chart(metric_data, img_path=f'image/{title} (Loss).png')
