# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 17:42
# @Author  : zhuo & zdy
# @github   : iezhuozhuo


def init_wt_normal(wt, trunc_norm_init_std=1e-4):
    wt.data.normal_(std=trunc_norm_init_std)


def init_rnn_wt(rnn_cell, rand_unif_init_mag=0.02):
    for names in rnn_cell._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn_cell, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn_cell, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear, trunc_norm_init_std=1e-4):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)
