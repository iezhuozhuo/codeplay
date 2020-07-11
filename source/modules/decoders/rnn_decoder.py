#!/usr/bin/env python
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/rnn_decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.attention import Attention, PointerAttention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.rnn_input_size += self.memory_size
            self.out_input_size += self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
        )
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        rnn_input_list = []
        out_input_list = []
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        rnn_input_list.append(input)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            rnn_input_list.append(feature)

        if self.attn_mode is not None:
            attn_memory = state.attn_memory
            attn_mask = state.attn_mask
            query = hidden[-1].unsqueeze(1)
            weighted_context, attn = self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=attn_mask)
            rnn_input_list.append(weighted_context)
            out_input_list.append(weighted_context)
            output.add(attn=attn)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        out_input_list.append(rnn_output)

        out_input = torch.cat(out_input_list, dim=-1)
        state.hidden = new_hidden

        if is_training:
            return out_input, state, output
        else:
            log_prob = self.output_layer(out_input)
            return log_prob, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, _ = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)

        log_probs = self.output_layer(out_inputs)
        return log_probs, state


class PointerDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 vocab_size,
                 embedder=None,
                 num_layers=1,
                 dropout=0.0,
                 pointer_gen=True):
        super(PointerDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.dropout = dropout
        self.pointer_gen = pointer_gen
        self.vocab_size = vocab_size

        self.attention_network = PointerAttention(self.hidden_size)

        # [input_embed, c_context]
        self.x_context = nn.Linear(self.hidden_size * 2 + self.input_size, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        # init_lstm_wt(self.lstm)

        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(self.hidden_size * 4 + self.input_size, 1)

        # p_vocab
        self.out1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.out2 = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, y_t, s_t_pre, encoder_outputs, encoder_feature, enc_padding_mask,
                h_context_pre, extra_zeros, article_ids_extend_vocab, coverage, step):
        """
        :param y_t: 每一时间步输入 id
        :param s_t_pre: LSTM 上一个时间步的隐状态包含 c,h
        :param encoder_outputs: 编码器的隐状态
        :param encoder_feature: 编码器的线型变换 W_h * h
        :param enc_padding_mask: 编码器的 mask
        :param c_context_pre: 上一个时间步的上下文向量 c_context
        :param extra_zeros:
        :param article_ids_extend_vocab: 虽然不在 article+summary 上的词表中，但是在 article 的原文中，oov可以copy[oov使用n_vocab+idx表示]
        :param coverage: coverage mechanism 防止 Repitition 类似机翻中解决过翻译/漏翻译 当前步所有 attention_weight 和 coverage_pre加和
        :param step: 译码时间步
        :return:
        """

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_pre
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_size),
                                 c_decoder.view(-1, self.hidden_size)), 1)  # B x 2*hidden_dim
            # attention return h_context, attn_weight, coverage
            h_context, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_embd = self.embedder(y_t)
        x = self.x_context(torch.cat((h_context_pre, y_t_embd), 1))
        self.lstm.flatten_parameters()
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_pre)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_size),
                             c_decoder.view(-1, self.hidden_size)), 1)  # B x 2*hidden_dim
        h_context, attn_weight, coverage_next = self.attention_network(
            s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            # p_gen = sigmoid(W_h*h_context + W_s*s + W_x*x)
            p_gen_input = torch.cat((h_context, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hidden_size), h_context), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_weight

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros.float()], 1)
            article_ids_extend_vocab_ = article_ids_extend_vocab.index_select(1, torch.arange(0, attn_dist_.size(1), dtype=int).to(article_ids_extend_vocab.device))
            # 把得到的单词分布加到对应的位置上 article_ids_extend_vocab知道word_index
            final_dist = vocab_dist_.scatter_add(1, article_ids_extend_vocab_, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, h_context, attn_weight, p_gen, coverage
