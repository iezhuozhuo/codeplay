# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 21:01
# @Author  : zhuo & zdy
# @github   : iezhuozhuo

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


# TODO 重合可以删除
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


class PointerAttention(nn.Module):
    def __init__(self, hidden_size, is_coverage=True):
        super(PointerAttention, self).__init__()

        self.hidden_size = hidden_size
        self.is_coverage = is_coverage
        # attention
        if self.is_coverage:
            self.W_c = nn.Linear(1, self.hidden_size * 2, bias=False)
        self.W_s = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.V = nn.Linear(self.hidden_size * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, encoder_mask, coverage):
        """
        :param s_t_hat: 对于 LSTM 是c,h的按列拼接， GRU尚未实现 公式中的s_t
        :param encoder_outputs: attention机制中的 v
        :param encoder_feature: encoder_outputs 的线性变换  W_h*encoder_outputs
        :param encoder_mask: 输入的 mask
        :param coverage: 是否使用 coverage
        :return: 上下文向量，注意力权重， coverage向量
        """
        bsz, seq, dim = list(encoder_outputs.size())

        dec_fea = self.W_s(s_t_hat)  # B x 2*hid_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(bsz, seq, dim).contiguous()  # B x seq x 2*hid_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, dim)  # B * seq x 2*hid_dim
        att_features = encoder_feature + dec_fea_expanded  # B * seq x 2*hidden_dim

        if self.is_coverage:
            coverage = coverage.index_select(1, torch.arange(0, seq, dtype=int).to(coverage.device))
            coverage_input = coverage.view(-1, 1)  # B * seq x 1
            coverage_feature = self.W_c(coverage_input)  # B * seq x 2*hidden_dim
            # W_s*s_t + W_h*h + W_c*coverage
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * seq x 2*hidden_dim
        scores = self.V(e)  # B * seq x 1
        scores = scores.view(-1, seq)  # B x seq

        # attn_weight_ = F.softmax(scores, dim=1) * encoder_mask  # B x seq
        attn_weight_ = F.softmax(scores, dim=1) * encoder_mask.index_select(1, torch.arange(0, seq, dtype=int).to(encoder_mask.device))
        normalization_factor = attn_weight_.sum(1, keepdim=True)
        attn_weight = attn_weight_ / normalization_factor

        attn_weight = attn_weight.unsqueeze(1)  # B x 1 x seq
        h_context = torch.bmm(attn_weight, encoder_outputs)  # B x 1 x dim
        h_context = h_context.view(-1, self.hidden_size * 2)  # B x 2*hidden_dim

        attn_weight = attn_weight.view(-1, seq)  # B x seq

        if self.is_coverage:
            coverage = coverage.view(-1, seq)
            coverage = coverage + attn_weight
            if seq < encoder_mask.size(1):
                zeros = coverage.new_zeros(coverage.size(0), encoder_mask.size(1) - seq)
                coverage = torch.cat([coverage, zeros], dim=1)

        return h_context, attn_weight, coverage


class BidirectionalAttention(nn.Module):
    """Computing the soft attention between two sequence."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward."""
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        v2_v1_attn = F.softmax(
            similarity_matrix.masked_fill(
                v1_mask.unsqueeze(2), -1e-7), dim=1)
        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1), -1e-7), dim=2)

        attended_v1 = v1_v2_attn.bmm(v2)  # 使用v2表示的v1
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)  # 使用v1表示的v2

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2
