# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

"""Matching module."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class Matching(nn.Module):
    """
    Module that computes a matching matrix between samples in two tensors.
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to `True`, then the output of the dot product
        is the cosine proximity between the two samples.
    :param matching_type: the similarity function for matching
    Examples:
        >>> import torch
        >>> matching = Matching(matching_type='dot', normalize=True)
        >>> x = torch.randn(2, 3, 2)
        >>> y = torch.randn(2, 4, 2)
        >>> matching(x, y).shape
        torch.Size([2, 3, 4])
    """

    def __init__(self, normalize: bool = False, matching_type: str = 'dot'):
        """:class:`Matching` constructor."""
        super().__init__()
        self._normalize = normalize
        self._validate_matching_type(matching_type)
        self._matching_type = matching_type

    @classmethod
    def _validate_matching_type(cls, matching_type: str = 'dot'):
        valid_matching_type = ['dot', 'exact', 'mul', 'plus', 'minus', 'concat']
        if matching_type not in valid_matching_type:
            raise ValueError(f"{matching_type} is not a valid matching type, "
                             f"{valid_matching_type} expected.")

    def forward(self, x, y):
        """Perform attention on the input."""
        length_left = x.shape[1]
        length_right = y.shape[1]
        if self._matching_type == 'dot':
            if self._normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            return torch.einsum('bld,brd->blr', x, y)
        elif self._matching_type == 'exact':
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1)
            matching_matrix = (x == y)
            x = torch.sum(matching_matrix, dim=2, dtype=torch.float)
            y = torch.sum(matching_matrix, dim=1, dtype=torch.float)
            return x, y
        else:
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right, 1)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1, 1)
            if self._matching_type == 'mul':
                return x * y
            elif self._matching_type == 'plus':
                return x + y
            elif self._matching_type == 'minus':
                return x - y
            elif self._matching_type == 'concat':
                return torch.cat((x, y), dim=3)


class MatchingTensor(nn.Module):
    """
    Module that captures the basic interactions between two tensors.
    :param matching_dims: Word dimension of two interaction texts.
    :param channels: Number of word interaction tensor channels.
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.
    Examples:
        >>> from source.modules.matching import MatchingTensor
        >>> matching_dim = 5
        >>> matching_tensor = MatchingTensor(
        ...    matching_dim,
        ...    channels=4,
        ...    normalize=True,
        ...    init_diag=True
        ... )
    """

    def __init__(
        self,
        matching_dim: int,
        channels: int = 4,
        normalize: bool = True,
        init_diag: bool = True
    ):
        """:class:`MatchingTensor` constructor."""
        super().__init__()
        self._matching_dim = matching_dim
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag

        self.interaction_matrix = torch.empty(
            self._channels, self._matching_dim, self._matching_dim
        )
        if self._init_diag:
            self.interaction_matrix = self.interaction_matrix.uniform_(-0.05, 0.05)
            for channel_index in range(self._channels):
                self.interaction_matrix[channel_index].fill_diagonal_(0.1)
            self.interaction_matrix = nn.Parameter(self.interaction_matrix)
        else:
            self.interaction_matrix = nn.Parameter(self.interaction_matrix.uniform_())

    def forward(self, x, y):
        """
        The computation logic of MatchingTensor.
        :param inputs: two input tensors.
        """

        if self._normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)

        # output = [b, c, l, r]
        output = torch.einsum(
            'bld,cde,bre->bclr',
            x, self.interaction_matrix, y
        )
        return output


def mp_matching_func(v1, v2, w):
    """
    Basic mp_matching_func.
    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (num_psp, hidden_size)
    :return: (batch, seq_len, num_psp)
    """

    seq_len = v1.size(1)
    num_psp = w.size(0)

    # (1, 1, hidden_size, num_psp)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    # (batch, seq_len, hidden_size, num_psp)
    v1 = w * torch.stack([v1] * num_psp, dim=3)
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * num_psp, dim=3)
    else:
        v2 = w * torch.stack(
            [torch.stack([v2] * seq_len, dim=1)] * num_psp, dim=3)

    m = F.cosine_similarity(v1, v2, dim=2)

    return m


def mp_matching_func_pairwise(v1, v2, w):
    """
    Basic mp_matching_func_pairwise.
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (num_psp, hidden_size)
    :param num_psp
    :return: (batch, num_psp, seq_len1, seq_len2)
    """

    num_psp = w.size(0)

    # (1, num_psp, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)
    # (batch, num_psp, seq_len, hidden_size)
    v1, v2 = (w * torch.stack([v1] * num_psp, dim=1),
              w * torch.stack([v2] * num_psp, dim=1))
    # (batch, num_psp, seq_len, hidden_size->1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_psp, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_psp)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)

    return m


def mp_matching_attention(v1, v2):
    """
    Attention.
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """

    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm  # 和矩阵乘得到的维度相似如[2,2,1]*[2,1,2]->[2,2,2]，但是不是向量乘加和而是直接点乘不用相加

    return div_with_small_value(a, d)


def div_with_small_value(n, d, eps=1e-8):
    """
    Small values are replaced by 1e-8 to prevent it from exploding.
    :param n: tensor
    :param d: tensor
    :return: n/d: tensor
    """
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d