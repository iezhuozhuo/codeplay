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

        self.out = self.quickly_output_layer(
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

