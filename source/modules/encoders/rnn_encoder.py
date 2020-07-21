#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/rnn_encoder.py
"""

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from source.modules.initial_weight import init_rnn_wt


class LSTMEncoder(nn.Module):
    """
    A LSTM recurrent neural network encoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_hidden_size=None,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 output_type="seq2seq"):
        super(LSTMEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        if not rnn_hidden_size:
            assert hidden_size % self.num_directions == 0
        else:
            assert rnn_hidden_size == hidden_size
        self.rnn_hidden_size = rnn_hidden_size or hidden_size // 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.output_type = output_type

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           dropout=self.dropout if self.num_layers > 1 else 0,
                           bidirectional=self.bidirectional)
        init_rnn_wt(self.rnn)

        if self.output_type == "seq2seq":
            self.W_h = nn.Linear(
                self.rnn_hidden_size * self.num_directions, self.rnn_hidden_size * self.num_directions, bias=False
            )

    def forward(self, inputs, hidden=None):
        """
        forward
        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        num_valid = None
        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()  # 当batch不足batch_size
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        self.rnn.flatten_parameters()
        # outputs (batch, seq_len, num_directions * rnn_hidden_size)
        # h (num_layers * num_directions, batch, rnn_hidden_size)
        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        h, c = last_hidden
        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.rnn_hidden_size * self.num_directions)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros_h = last_hidden[0].new_zeros(
                    self.num_layers, batch_size - num_valid, self.rnn_hidden_size * self.num_directions)
                h = torch.cat([last_hidden[0], zeros_h], dim=1)
                zeros_c = last_hidden[1].new_zeros(
                    self.num_layers, batch_size - num_valid, self.rnn_hidden_size * self.num_directions)
                c = torch.cat([last_hidden[1], zeros_c], dim=1)
                last_hidden = (h, c)

            h, c = last_hidden
            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            h = h.index_select(1, inv_indices)
            c = c.index_select(1, inv_indices)

        if self.output_type == "seq2seq":
            encoder_feature = outputs.view(-1, self.num_directions * self.rnn_hidden_size)
            encoder_feature = self.W_h(encoder_feature)
            return outputs, encoder_feature, (h, c)

        return outputs, (h, c)

    def _bridge_bidirectional_hidden(self, hidden):
        """
        hidden is the last-hide
        the bidirectional hidden is (num_layers * num_directions, batch_size, rnn_hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * rnn_hidden_size)
        """
        h, c = hidden

        num_layers = h.size(0) // self.num_directions
        _, batch_size, rnn_hidden_size = h.size()
        h = h.view(num_layers, self.num_directions, batch_size, rnn_hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, rnn_hidden_size * self.num_directions)
        c = c.view(num_layers, self.num_directions, batch_size, rnn_hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, rnn_hidden_size * self.num_directions)

        return (h, c)


class GRUEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 rnn_hidden_size=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(GRUEncoder, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        if not rnn_hidden_size:
            assert hidden_size % self.num_directions == 0
        else:
            assert rnn_hidden_size == hidden_size
        self.rnn_hidden_size = rnn_hidden_size or hidden_size // 2

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()  # 当batch不足batch_size
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        self.rnn.flatten_parameters()
        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.rnn_hidden_size * self.num_directions)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.rnn_hidden_size * self.num_directions)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        hidden is the last-hide
        the bidirectional hidden is (num_layers * num_directions, batch_size, rnn_hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * rnn_hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)


# old vision gru based
class RNNEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):
        """
        forward
        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()  # 当batch不足batch_size
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        self.rnn.flatten_parameters()
        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        hidden is the last-hide
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size) \
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)


class HRNNEncoder(nn.Module):
    """
    HRNNEncoder
    """

    def __init__(self,
                 sub_encoder,
                 hiera_encoder):
        super(HRNNEncoder, self).__init__()
        self.sub_encoder = sub_encoder
        self.hiera_encoder = hiera_encoder

    def forward(self, inputs, features=None, sub_hidden=None, hiera_hidden=None,
                return_last_sub_outputs=False):
        """
        inputs: Tuple[Tensor(batch_size, max_hiera_len, max_sub_len), 
                Tensor(batch_size, max_hiera_len)]
        """
        indices, lengths = inputs
        batch_size, max_hiera_len, max_sub_len = indices.size()
        hiera_lengths = lengths.gt(0).long().sum(dim=1)

        # Forward of sub encoder
        indices = indices.view(-1, max_sub_len)
        sub_lengths = lengths.view(-1)
        sub_enc_inputs = (indices, sub_lengths)
        sub_outputs, sub_hidden = self.sub_encoder(sub_enc_inputs, sub_hidden)
        sub_hidden = sub_hidden[-1].view(batch_size, max_hiera_len, -1)

        if features is not None:
            sub_hidden = torch.cat([sub_hidden, features], dim=-1)

        # Forward of hiera encoder
        hiera_enc_inputs = (sub_hidden, hiera_lengths)
        hiera_outputs, hiera_hidden = self.hiera_encoder(
            hiera_enc_inputs, hiera_hidden)

        if return_last_sub_outputs:
            sub_outputs = sub_outputs.view(
                batch_size, max_hiera_len, max_sub_len, -1)
            last_sub_outputs = torch.stack(
                [sub_outputs[b, l - 1] for b, l in enumerate(hiera_lengths)])
            last_sub_lengths = torch.stack(
                [lengths[b, l - 1] for b, l in enumerate(hiera_lengths)])
            max_len = last_sub_lengths.max()
            last_sub_outputs = last_sub_outputs[:, :max_len]
            return hiera_outputs, hiera_hidden, (last_sub_outputs, last_sub_lengths)
        else:
            return hiera_outputs, hiera_hidden, None
