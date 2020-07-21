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