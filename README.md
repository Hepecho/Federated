# Lab 5.1: Distributed Attack and Defense
## Introduction
this project is a reproduction code for Distributed Deep Learning, 
concening simple attack like Label Flipping and Bit Flipping (from [this paper](https://arxiv.org/abs/1805.10032)), 
and defence aggregation method [Krum](https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf).
## Usage
### Dataset
please download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset in directory `./data/cifao10/`,
or automatically download when running code
### Example
Before attempting any further experiments, please confirm whether the configuration is correct in `./src/Config.py`

if you'd like to test Lable Flipping attack, run:
```
python src/main.py --action 1
```
The result will be output in the console and saved in `./log/` and model will be saved in `./checkpoint`

If you need to test other experiments, please check the `action` parameter in `./src/main.py`
