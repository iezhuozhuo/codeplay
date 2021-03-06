# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 21:01
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import os
import time
import random
import numpy as np
import argparse
import logging

import torch


def get_model_parameters_num(model):
    # 记录模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    return num_parameters


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def checkoutput_and_setcuda(args):
    args.output_dir = os.path.join(args.output_dir, args.model_type)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    return args


def init_logger(args=None):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args != None:
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank,
            args.device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )
    return logger


logger = init_logger()
def timer(func):
    """耗时装饰器，计算函数运行时长"""
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        logger.info(f"Cost time: {cost} s")
        return r

    return wrapper


# 定制monitor metric 最简单的monitor loss
# if EarlyStopping.early_stop break
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, is_decreased_valid_metric=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        TODO: 利用delta (float)
        """
        self.patience = patience
        self.is_decreased_valid_metric = is_decreased_valid_metric
        self.counter = 0
        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.early_stop = False
        self.is_best = False
        self.delta = delta

    def __call__(self, cur_valid_metric):

        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric
        if is_best:
            self.best_valid_metric = cur_valid_metric
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

        if self.counter >= self.patience:
            self.early_stop = True


class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = []
        for vs in zip(*self.values()):
            pack = Pack(zip(self.keys(), vs))
            pack_list.append(pack)
        return pack_list

    # def cuda(self, device=None):
    #     """
    #     cuda
    #     """
    #     pack = Pack()
    #     for k, v in self.items():
    #         if isinstance(v, tuple):
    #             pack[k] = tuple(x.cuda(device) for x in v)
    #         else:
    #             pack[k] = v.cuda(device)
    #     return pack

    # FIX多GPU
    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.to(device) for x in v)
            else:
                pack[k] = v.to(device)
        return pack


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):
    """
    list2tensor
    """
    size = max_lens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


# def list2tensor(X, max_len=None):
#     sizes = max_lens(X)
#
#     if len(sizes) == 1:
#         tensor = torch.tensor(X)
#         return tensor
#
#     if max_len is not None:
#         assert max_len >= sizes[-1]
#         sizes[-1] = max_len
#
#     tensor = torch.zeros(sizes, dtype=torch.long)
#     lengths = torch.zeros(sizes[:-1], dtype=torch.long)
#     if len(sizes) == 2:
#         for i, x in enumerate(X):
#             l = len(x)
#             tensor[i, :l] = torch.tensor(x)
#             lengths[i] = l
#     else:
#         for i, xs in enumerate(X):
#             for j, x in enumerate(xs):
#                 l = len(x)
#                 tensor[i, j, :l] = torch.tensor(x)
#                 lengths[i, j] = l
#
#     return tensor, lengths


# def one_hot(indice, vocab_size):
#     T = torch.zeros(*indice.size(), vocab_size).type_as(indice).float()
#     T = T.scatter(-1, indice.unsqueeze(-1), 1)
#     return T

def one_hot(indice, num_classes):
    """
    one_hot
    """
    I = torch.eye(num_classes).to(indice.device)
    T = I[indice]
    return T


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')




if __name__ == '__main__':
    X = [1, 2, 3]
    print(X)
    print(list2tensor(X))
    X = [X, [2, 3]]
    print(X)
    print(list2tensor(X))
    X = [X, [[1, 1, 1, 1, 1]]]
    print(X)
    print(list2tensor(X))

    data_list = [{'src': [1, 2, 3], 'tgt': [1, 2, 3, 4]},
                 {'src': [2, 3], 'tgt': [1, 2, 4]}]
    batch = Pack()
    for key in data_list[0].keys():
        batch[key] = list2tensor([x[key] for x in data_list], 8)
    print(batch)
