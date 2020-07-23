# -*- coding: utf-8 -*-
# @Time    : 2020/7/17 14:25
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.activate import parse_activation
from source.modules.linears import quickly_output_layer, quickly_multi_layer_perceptron_layer
from source.modules.matching import Matching
from source.modules.encoders.rnn_encoder import LSTMEncoder


class ArcI(nn.Module):
    def __init__(self,
                 embedd=None,
                 left_filters=None,
                 left_kernel_sizes=None,
                 left_pool_sizes=None,
                 left_length=32,
                 right_filters=None,
                 right_kernel_sizes=None,
                 right_pool_sizes=None,
                 right_length=32,
                 conv_activation_func='relu',
                 dropout_rate=0.5
                 ):
        """
        :param embedd:
        :param left_filters:
        :param left_kernel_sizes:
        :param left_pool_sizes:
        :param left_length:  用于计算 pooling 之后的长度 便于使用output计算match score
        :param right_filters:
        :param right_kernel_sizes:
        :param right_pool_sizes:
        :param right_length: 用于计算 pooling 之后的长度 便于使用output计算match score
        :param conv_activation_func:
        :param dropout_rate: 基本上没有用
        """
        super(ArcI, self).__init__()

        if embedd is None:
            raise Exception("The embdding layer is None")

        if right_pool_sizes is None:
            right_pool_sizes = [2]
        if right_kernel_sizes is None:
            right_kernel_sizes = [3]
        if right_filters is None:
            right_filters = [32]
        if left_pool_sizes is None:
            left_pool_sizes = [2]
        if left_kernel_sizes is None:
            left_kernel_sizes = [3]
        if left_filters is None:
            left_filters = [32]

        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim
        self.left_filters = left_filters
        self.left_kernel_sizes = left_kernel_sizes
        self.left_pool_sizes = left_pool_sizes
        self.right_filters = right_filters
        self.right_kernel_sizes = right_kernel_sizes
        self.right_pool_sizes = right_pool_sizes
        self.conv_activation_func = conv_activation_func
        self.dropout_rate = dropout_rate
        self.left_length = left_length
        self.right_length = right_length

        self.build()

    def forward(self, inputs):
        input_left, input_right = inputs['text_a'], inputs['text_b']

        # shape = [B, D, L]
        # shape = [B, D, R]
        embed_left = self.embedding(input_left).transpose(1, 2)
        embed_right = self.embedding(input_right).transpose(1, 2)

        # Convolution
        # shape = [B, F, L // P]
        # shape = [B, F, R // P]
        conv_left = self.conv_left(embed_left)
        conv_right = self.conv_right(embed_right)

        # shape = [B, F * (L // P)]
        # shape = [B, F * (R // P)]
        rep_left = torch.flatten(conv_left, start_dim=1)
        rep_right = torch.flatten(conv_right, start_dim=1)

        # shape = [B, F * (L // P) + F * (R // P)]
        concat = self.dropout(torch.cat((rep_left, rep_right), dim=1))

        # shape = [B, *]
        dense_output = self.mlp(concat)

        out = self.out(dense_output)
        return out

    def build(self):
        left_in_channels = [
            self.embedding_dim,
            *self.left_filters[:-1]
        ]
        right_in_channels = [
            self.embedding_dim,
            *self.right_filters[:-1]
        ]
        activation = parse_activation(self.conv_activation_func)
        conv_left = [
            self.quickly_conv_pool_back(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(left_in_channels,
                                      self.left_filters,
                                      self.left_kernel_sizes,
                                      self.left_pool_sizes)
        ]
        conv_right = [
            self.quickly_conv_pool_back(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(right_in_channels,
                                      self.right_filters,
                                      self.right_kernel_sizes,
                                      self.right_pool_sizes)
        ]
        self.conv_left = nn.Sequential(*conv_left)
        self.conv_right = nn.Sequential(*conv_right)

        left_length = self.left_length
        right_length = self.right_length
        for ps in self.left_pool_sizes:
            left_length = left_length // ps
        for ps in self.right_pool_sizes:
            right_length = right_length // ps

        self.mlp = quickly_multi_layer_perceptron_layer(
            left_length * self.left_filters[-1] + (
                    right_length * self.right_filters[-1])
        )

        self.out = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=64
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)

    @classmethod
    def quickly_conv_pool_back(cls,
                               in_channels: int,
                               out_channels: int,
                               kernel_size: int,
                               activation: nn.Module,
                               pool_size: int,
                               ) -> nn.Module:
        return nn.Sequential(
            nn.ConstantPad1d((0, kernel_size - 1), 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool1d(kernel_size=pool_size)
        )


class ArcII(nn.Module):
    def __int__(self,
                embedd=None,
                left_length=32,
                right_length=32,
                kernel_1d_count=32,
                kernel_1d_size=3,
                kernel_2d_count=None,
                kernel_2d_size=None,
                pool_2d_size=None,
                conv_activation_func='relu',
                dropout_rate=0.5
                ):
        super(ArcII, self).__int__()

        if embedd is None:
            raise Exception("The embdding layer is None")
        if pool_2d_size is None:
            pool_2d_size = [(2, 2)]
        if kernel_2d_size is None:
            kernel_2d_size = [(3, 3)]
        if kernel_2d_count is None:
            kernel_2d_count = [32]

        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim
        self.left_length = left_length
        self.right_length = right_length
        self.kernel_1d_count = kernel_1d_count
        self.kernel_1d_size = kernel_1d_size
        self.kernel_2d_count = kernel_2d_count
        self.kernel_2d_size = kernel_2d_size
        self.pool_2d_size = pool_2d_size
        self.conv_activation_func = conv_activation_func
        self.dropout_rate = dropout_rate

        self.buid()

    def buid(self):
        # Phrase level representations
        self.conv1d_left = nn.Sequential(
            nn.ConstantPad1d((0, self.kernel_1d_size - 1), 0),
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.kernel_1d_count,
                kernel_size=self.kernel_1d_size
            )
        )
        self.conv1d_right = nn.Sequential(
            nn.ConstantPad1d((0, self.kernel_1d_size - 1), 0),
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=self.kernel_1d_count,
                kernel_size=self.kernel_1d_size
            )
        )

        # Interaction
        self.matching = Matching(matching_type='plus')

        # Build conv
        activation = parse_activation(self.conv_activation_func)
        in_channel_2d = [
            self.kernel_1d_count,
            *self.kernel_2d_count[:-1]
        ]
        conv2d = [
            self._make_conv_pool_block(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(in_channel_2d,
                                      self.kernel_2d_count,
                                      self.kernel_2d_size,
                                      self.pool_2d_size)
        ]
        self.conv2d = nn.Sequential(*conv2d)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        left_length = self.left_length
        right_length = self.right_length
        for ps in self.pool_2d_size:
            left_length = left_length // ps[0]
        for ps in self.pool_2d_size:
            right_length = right_length // ps[1]

        # Build output
        self.out = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=left_length * right_length * self.kernel_2d_count[-1]
        )

    def forward(self, inputs):
        input_left, input_right = inputs['text_a'], inputs['text_b']
        # Process left and right input.
        # shape = [B, D, L]
        # shape = [B, D, R]
        embed_left = self.embedding(input_left).transpose(1, 2)
        embed_right = self.embedding(input_right).transpose(1, 2)

        # shape = [B, L, F1]
        # shape = [B, R, F1]
        conv1d_left = self.conv1d_left(embed_left).transpose(1, 2)
        conv1d_right = self.conv1d_right(embed_right).transpose(1, 2)

        # Compute matching signal
        # shape = [B, L, R, F1]
        embed_cross = self.matching(conv1d_left, conv1d_right)

        # Convolution
        # shape = [B, F2, L // P, R // P]
        conv = self.conv2d(embed_cross.permute(0, 3, 1, 2))

        # shape = [B, F2 * (L // P) * (R // P)]
        embed_flat = self.dropout(torch.flatten(conv, start_dim=1))

        # shape = [B, *]
        out = self.out(embed_flat)
        return out

    @classmethod
    def quickly_conv_pool_back(cls,
                               in_channels: int,
                               out_channels: int,
                               kernel_size: tuple,
                               activation: nn.Module,
                               pool_size: tuple,
                               ) -> nn.Module:
        """Make conv pool block."""
        return nn.Sequential(
            # Same padding
            nn.ConstantPad2d(
                (0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool2d(kernel_size=pool_size)
        )


class MVLSTM(nn.Module):
    def __init__(self,
                 embedd=None,
                 hidden_size=128,
                 num_layers=1,
                 top_k=5,
                 mlp_num_layers=2,
                 mlp_num_units=64,
                 mlp_num_fan_out=64,
                 activation_func='relu',
                 dropout_rate=0.5,
                 bidirectional=True
                 ):
        super(MVLSTM, self).__init__()
        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.top_k = top_k
        self.mlp_num_layers = mlp_num_layers
        self.mlp_num_units = mlp_num_units
        self.mlp_num_fan_out = mlp_num_fan_out
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.build()

    def build(self):
        self.left_bilstm = LSTMEncoder(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            rnn_hidden_size=self.hidden_size,
            embedder=self.embedding,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate,
            output_type="encode")

        self.right_bilstm = LSTMEncoder(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            rnn_hidden_size=self.hidden_size,
            embedder=self.embedding,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate,
            output_type="encode")

        self.mlp = quickly_multi_layer_perceptron_layer(
            in_features=self.top_k,
            mlp_num_layers=self.mlp_num_layers,
            mlp_num_units=self.mlp_num_units,
            mlp_num_fan_out=self.mlp_num_fan_out,
            mlp_activation_func=self.activation_func
        )

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.out = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=self.mlp_num_fan_out,
            out_activation_func=self.activation_func
        )

    def forward(self, inputs):
        inputs_left, inputs_left_len, inputs_right, inputs_right_len = \
            inputs["text_a"], inputs["text_a_len"], inputs["text_b"], inputs["text_b_len"]

        # Bi-directional LSTM
        # shape = [B, L, 2 * H]
        # shape = [B, R, 2 * H]
        rep_query, _ = self.left_bilstm((inputs_left, inputs_left_len))
        rep_doc, _ = self.right_bilstm((inputs_right, inputs_right_len))

        # Top-k matching
        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(rep_query, p=2, dim=-1),
            F.normalize(rep_doc, p=2, dim=-1)
        )
        # shape = [B, L * R]
        matching_signals = torch.flatten(matching_matrix, start_dim=1)
        # shape = [B, K]
        matching_topk = torch.topk(
            matching_signals,
            k=self.top_k,
            dim=-1,
            sorted=True
        )[0]

        # shape = [B, *]
        dense_output = self.mlp(matching_topk)

        # shape = [B, *]
        out = self.out(self.dropout(dense_output))
        return out


class MatchPyramid(nn.Module):
    def __init__(self,
                 embedd=None,
                 kernel_count=None,
                 kernel_size=None,
                 activation_func='relu',
                 dpool_size=None):
        super(MatchPyramid, self).__init__()
        if embedd is None:
            raise Exception("The embdding layer is None")
        if kernel_count is None:
            kernel_count = [32]
        elif isinstance(kernel_count, int):
            kernel_count = [kernel_count]
        if kernel_size is None:
            kernel_size = [[3, 3]]
        if dpool_size is None:
            dpool_size = [3, 10]

        self.embedding = embedd
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.activation_func = activation_func
        self.dpool_size = dpool_size

        self.build()

    def build(self):
        # Interaction
        self.matching = Matching(matching_type='dot')

        # Build conv
        activation = parse_activation(self.activation_func)
        in_channel_2d = [
            1,
            *self.kernel_count[:-1]
        ]
        conv2d = [
            self.quickly_conv_pool_block(ic, oc, ks, activation)
            for ic, oc, ks, in zip(in_channel_2d,
                                   self.kernel_count,
                                   self.kernel_size)
        ]
        self.conv2d = nn.Sequential(*conv2d)

        # Dynamic Pooling
        self.dpool_layer = nn.AdaptiveAvgPool2d(self.dpool_size)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        left_length = self.dpool_size[0]
        right_length = self.dpool_size[1]

        # Build output
        self.out = self.quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=left_length * right_length * self.kernel_count[-1]
        )

    def forward(self, inputs):
        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_a'], inputs['text_b']

        # Process left and right input.
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        # Compute matching signal
        # shape = [B, 1, L, R]
        embed_cross = self.matching(embed_left, embed_right).unsqueeze(dim=1)

        # Convolution
        # shape = [B, F, L, R]
        conv = self.conv2d(embed_cross)

        # Dynamic Pooling
        # shape = [B, F, P1, P2]
        embed_pool = self.dpool_layer(conv)

        # shape = [B, F * P1 * P2]
        embed_flat = self.dropout(torch.flatten(embed_pool, start_dim=1))

        # shape = [B, *]
        out = self.out(embed_flat)
        return out


    @classmethod
    def quickly_conv_pool_block(
            cls,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple,
            activation: nn.Module
    ) -> nn.Module:
        """Make conv pool block."""
        return nn.Sequential(
            # Same padding
            nn.ConstantPad2d(
                (0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation
        )

class BiMPM_Model(nn.Module):
    def __init__(self,
                 embedd=None,
                 hidden_size=128,
                 num_layers=1,
                 top_k=5,
                 mlp_num_layers=2,
                 mlp_num_units=64,
                 mlp_num_fan_out=64,
                 activation_func='relu',
                 dropout_rate=0.5,
                 bidirectional=True,
                 l = 5
                 ):
        super(BiMPM_Model, self).__init__()
        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.top_k = top_k
        self.mlp_num_layers = mlp_num_layers
        self.mlp_num_units = mlp_num_units
        self.mlp_num_fan_out = mlp_num_fan_out
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.l = l




        self.context_LSTM  = LSTMEncoder(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            rnn_hidden_size=self.hidden_size,
            embedder=self.embedding,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate,
            output_type="encode")


        self.aggregation_LSTM  = LSTMEncoder(
            input_size=self.l * 8,
            hidden_size=self.hidden_size,
            rnn_hidden_size=self.hidden_size,
            embedder=None,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate,
            output_type="encode")

        self.dropout = nn.Dropout(p=self.dropout_rate)





        for i in range(1, 9):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.l, self.hidden_size)))

        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal(w)

        self.pred_fc1 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.pred_fc2 = nn.Linear(self.hidden_size * 2, 2)


    # article_ids, article_len, article_mask,
    # summary_input_ids, summary_len, summary_taget_ids, summary_mask,
    # article_ids_extend_vocab = None, article_oovs = None, extra_zeros = None,
    def forward(self, inputs):
        inputs_left, inputs_left_len, inputs_right, inputs_right_len = \
            inputs["text_a"], inputs["text_a_len"], inputs["text_b"], inputs["text_b_len"]

        # Bi-directional LSTM
        # shape = [B, L, 2 * H]
        # shape = [B, R, 2 * H]
        rep_query, _ = self.context_LSTM((inputs_left, inputs_left_len))
        rep_doc, _ = self.context_LSTM((inputs_right, inputs_right_len))

        rep_query = self.dropout(rep_query)
        rep_doc = self.dropout(rep_doc)


        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(rep_query, self.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(rep_doc, self.hidden_size, dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = self.mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = self.mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = self.mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = self.mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = self.mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = self.mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)


        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = self.attention(con_p_fw, con_h_fw)
        att_bw = self.attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = self.div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = self.div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = self.div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = self.div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, l)
        mv_p_att_mean_fw = self.mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = self.mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = self.mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = self.mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)


        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, l)
        mv_p_att_max_fw = self.mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = self.mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = self.mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = self.mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.hidden_size * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x


    def attention(self, v1, v2):
        """
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
        d = v1_norm * v2_norm

        return self.div_with_small_value(a, d)

    def mp_matching_func(self, v1, v2, w):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l)
        """
        seq_len = v1.size(1)

        # Trick for large memory requirement
        """
        if len(v2.size()) == 2:
            v2 = torch.stack([v2] * seq_len, dim=1)
        m = []
        for i in range(self.l):
            # v1: (batch, seq_len, hidden_size)
            # v2: (batch, seq_len, hidden_size)
            # w: (1, 1, hidden_size)
            # -> (batch, seq_len)
            m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))
        # list of (batch, seq_len) -> (batch, seq_len, l)
        m = torch.stack(m, dim=2)
        """

        # (1, 1, hidden_size, l)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        v1 = w * torch.stack([v1] * self.l, dim=3)
        if len(v2.size()) == 3:
            v2 = w * torch.stack([v2] * self.l, dim=3)
        else:
            v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

        m = F.cosine_similarity(v1, v2, dim=2)

        return m

    def mp_matching_func_pairwise(self, v1, v2, w):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l, seq_len1, seq_len2)
        """

        # Trick for large memory requirement
        """
        m = []
        for i in range(self.l):
            # (1, 1, hidden_size)
            w_i = w[i].view(1, 1, -1)
            # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
            v1, v2 = w_i * v1, w_i * v2
            # (batch, seq_len, hidden_size->1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True)
            # (batch, seq_len1, seq_len2)
            n = torch.matmul(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm.permute(0, 2, 1)
            m.append(div_with_small_value(n, d))
        # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
        m = torch.stack(m, dim=3)
        """

        # (1, l, 1, hidden_size)
        w = w.unsqueeze(0).unsqueeze(2)
        # (batch, l, seq_len, hidden_size)
        v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
        # (batch, l, seq_len, hidden_size->1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)

        # (batch, l, seq_len1, seq_len2)
        n = torch.matmul(v1, v2.transpose(2, 3))
        d = v1_norm * v2_norm.transpose(2, 3)

        # (batch, seq_len1, seq_len2, l)
        m = self.div_with_small_value(n, d).permute(0, 2, 3, 1)

        return m

    def div_with_small_value(self, n, d, eps=1e-8):
        # too small values are replaced by 1e-8 to prevent it from exploding.
        d = d * (d > eps).float() + eps * (d <= eps).float()
        return n / d