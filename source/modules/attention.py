#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from source.utils.misc import sequence_mask


class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 return_attn_only=False,
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.project = project

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = "Attention({}, {}".format(self.query_size, self.memory_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        memory: Tensor(batch_size, memory_length, memory_size)
        mask: Tensor(batch_size, memory_length)
        """
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            # (batch_size, query_length, memory_size)
            key = self.linear_query(query)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            # (batch_size, query_length, memory_length, hidden_size)
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            # (batch_size, query_length, memory_length)
            attn = self.v(key).squeeze(-1)

        if mask is not None:
            # (batch_size, query_length, memory_length)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        # (batch_size, query_length, memory_length)
        weights = self.softmax(attn)
        if self.return_attn_only:
            return weights

        # (batch_size, query_length, memory_size)
        weighted_memory = torch.bmm(weights, memory)

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            return weighted_memory, weights


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
