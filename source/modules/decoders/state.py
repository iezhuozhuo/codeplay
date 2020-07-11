# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 21:01
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import  torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.initial_weight import init_linear_wt


class DecoderState(object):
    """
    State of Decoder.
    """

    def __init__(self, hidden=None, **kwargs):
        """
        hidden: Tensor(num_layers, batch_size, hidden_size)
        """
        if hidden is not None:
            self.hidden = hidden
        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    def get_batch_size(self):
        """
        get_batch_size
        """
        if self.hidden is not None:
            return self.hidden.size(1)
        else:
            return next(iter(self.__dict__.values())).size(0)

    def size(self):
        """
        size
        """
        sizes = {k: v.size() for k, v in self.__dict__.items()}
        return sizes

    def slice_select(self, stop):
        """
        slice_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                kwargs[k] = v[:, :stop].clone()
            else:
                kwargs[k] = v[:stop]
        return DecoderState(**kwargs)

    def index_select(self, indices):
        """
        index_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == 'hidden':
                kwargs[k] = v.index_select(1, indices)
            else:
                kwargs[k] = v.index_select(0, indices)
        return DecoderState(**kwargs)

    def mask_select(self, mask):
        """
        mask_select
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                kwargs[k] = v[:, mask]
            else:
                kwargs[k] = v[mask]
        return DecoderState(**kwargs)

    def _inflate_tensor(self, X, times):
        """
        inflate X from shape (batch_size, ...) to shape (batch_size*times, ...)
        for first decoding of beam search
        """
        sizes = X.size()

        if X.dim() == 1:
            X = X.unsqueeze(1)

        repeat_times = [1] * X.dim()
        repeat_times[1] = times
        X = X.repeat(*repeat_times).view(-1, *sizes[1:])
        return X

    def inflate(self, times):
        """
        inflate
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            if k == "hidden":
                num_layers, batch_size, _ = v.size()
                kwargs[k] = v.repeat(1, 1, times).view(
                    num_layers, batch_size * times, -1)
            else:
                kwargs[k] = self._inflate_tensor(v, times)
        return DecoderState(**kwargs)


class ReduceState(nn.Module):
    def __init__(self, hidden_size, bidirectional=True):
        super(ReduceState, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.reduce_h = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h_in, c_in = hidden  # # [layer, batch, hidden_dim*2]

        hidden_reduced_h = F.relu(self.reduce_h(h_in[-1]))  # [batch, hidden_dim]
        hidden_reduced_c = F.relu(self.reduce_c(c_in[-1]))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = [1, batch, hidden_dim]

