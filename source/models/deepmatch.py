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
from source.modules.encoders.rnn_encoder import LSTMEncoder, GRUEncoder


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
        """
        :param embedd:
        :param left_length:
        :param right_length:
        :param kernel_1d_count:
        :param kernel_1d_size:
        :param kernel_2d_count:
        :param kernel_2d_size:
        :param pool_2d_size:
        :param conv_activation_func:
        :param dropout_rate:
        :return:
        """
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
        self.out = self.quickly_output_layer(
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
                 top_k=50,
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
    

# TODO matchSRNN
class MatchSRNN(nn.Module):
    def __int__(self):
        super(MatchSRNN, self).__int__()


# MwAN
class MwAN(nn.Module):
    def __init__(self,
                 embedd=None,
                 hidden_size=128,
                 num_layers=1,
                 activation_func='relu',
                 dropout_rate=0.5,
                 bidirectional=True
                 ):
        super().__init__()
        """
        v1版本，没有忠于原文，仅实现了四种注意力机制。
        原文4.4部分/4.5部分 门控控制输入没有完成。
        """
        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim

        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        self.p_encoder = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,
                                bidirectional=self.bidirectional)
        self.c_encoder = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,
                                bidirectional=self.bidirectional)
        # Multi-Way Attention
        # concat attention
        self.Wc1 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Wc2 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Vc = nn.Linear(self.hidden_size, 1, bias=False)

        # Bilinear Attention
        self.Wb = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)

        # Dot Attention :
        self.Wd = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Vd = nn.Linear(self.hidden_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Vm = nn.Linear(self.hidden_size, 1, bias=False)

        # gate weight
        # self.Wgc = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgd = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgb = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgm = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)

        # 非原文部分
        self.Ws = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Vs = nn.Linear(self.hidden_size, 1, bias=False)

        self.gru_agg = nn.GRU(6 * self.num_directions * self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)

        # predict layer
        self.Wp = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.Vp = nn.Linear(self.hidden_size, 1, bias=False)
        self.W1 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.W2 = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size, bias=False)
        self.V = nn.Linear(self.hidden_size, 1, bias=False)
        self.output = quickly_output_layer(
            task="classify", num_classes=2, in_features=self.num_directions * self.hidden_size)

        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        left, right = inputs["text_a"], inputs["text_b"]
        p_embedding = self.embedding(left)
        c_embedding = self.embedding(right)

        self.p_encoder.flatten_parameters()
        hp, _ = self.p_encoder(p_embedding)
        hp = F.dropout(hp, self.dropout_rate)
        self.c_encoder.flatten_parameters()
        hc, _ = self.c_encoder(c_embedding)
        hc = F.dropout(hc, self.dropout_rate)

        # B,L,D    B,R,D  concat attention
        _s1 = self.Wc1(hp).unsqueeze(1)  # B,1,L,D
        _s2 = self.Wc2(hc).unsqueeze(2)  # B,R,1,D
        attn_concat = F.softmax(self.Vc(torch.tanh(_s1 + _s2)).squeeze(), 2)
        concat_rep = attn_concat.bmm(hp)

        # billiner attention
        _s1 = self.Wb(hp).transpose(2, 1)
        attn_billiner = F.softmax(hc.bmm(_s1), 2)
        billiner_rep = attn_billiner.bmm(hp)

        # Dot Attention 扩充一维度是因为 输入的两个长度不一致，需要使用矩阵加减乘除的广播
        _s1 = hp.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        attn_dot = F.softmax(self.Vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze(), 2)
        dot_rep = attn_dot.bmm(hp)

        # minus attention
        attn_minus = F.softmax(self.Vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze(), 2)
        minus_rep = attn_minus.bmm(hp)

        _s1 = hc.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        attn_self = F.softmax(self.Vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze(), 2)
        self_rep = attn_self.bmm(hc)

        aggregation = torch.cat([hc, self_rep, concat_rep, dot_rep, billiner_rep, minus_rep], 2)
        self.gru_agg.flatten_parameters()
        aggregation_representation, _ = self.gru_agg(aggregation)

        # 比原文少一个 vp向量
        sj = self.Vp(torch.tanh(self.Wp(hp))).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(hp)

        # 原文12a~12c公式
        sj = F.softmax(self.V(self.W1(aggregation_representation) + self.W2(rp)).transpose(2, 1), 2)
        rc = sj.bmm(aggregation_representation)
        # 归一化
        # encoder_output = F.sigmoid(self.prediction(rc))
        output = self.output(rc.squeeze())
        return output