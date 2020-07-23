# -*- coding: utf-8 -*-
# @Time    : 2020/7/17 14:25
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.activate import parse_activation
from source.modules.linears import quickly_output_layer, quickly_multi_layer_perceptron_layer
from source.modules.matching import Matching, mp_matching_func, mp_matching_func_pairwise, mp_matching_attention, \
    div_with_small_value
from source.modules.encoders.rnn_encoder import LSTMEncoder, GRUEncoder, StackedBRNN
from source.modules.attention import BidirectionalAttention
from source.modules.embedder import CharacterEmbedding
from source.modules.densenet import DenseNet, SemanticComposite


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
        """
        paper link: https://arxiv.org/pdf/1604.04378.pdf
        """


# TODO MIX
class MIX(nn.Module):
    def __init__(self):
        super(MIX, self).__init__()
    """
    参考： https://github.com/hanzhenlei767/NLP_Learn/tree/4a780c08788e7b4184c7bf42c3db1423532637ba/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D
    """


# MwAN
class MwAN(nn.Module):
    def __init__(self,
                 args,
                 embedd=None,
                 num_layers=1,
                 activation_func='relu',
                 dropout_rate=0.5,
                 bidirectional=True
                 ):
        super().__init__()
        """
        paper link: https://www.ijcai.org/Proceedings/2018/0613.pdf
        v1版本使用 https://github.com/syw1996/Retrieval-Automatic-Comment-System/blob/master/model.py 注意力计算方式
        v2版本优化参考 https://github.com/bulong/MANM/blob/master/MANM.py 补充了 inter_agg 和 mix_agg
        """
        self.args = args
        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim

        self.hidden_size = args.hidden_size

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
        self.Wg = nn.Linear(2 * self.hidden_size * self.num_directions, 1, bias=False)
        # self.Wgc = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgd = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgb = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)
        # self.Wgm = nn.Linear(2 * self.hidden_size * self.num_directions, self.hidden_size * self.num_directions, bias=False)

        self.gru_inter_agg = nn.GRU(2 * self.num_directions * self.hidden_size, self.hidden_size, batch_first=True,
                                    bidirectional=True)

        # mix aggregation
        self.Wx = nn.Linear(self.hidden_size * self.num_directions, 1, bias=False)
        self.vx = nn.Parameter(torch.randn(self.args.max_seq_length, 4))
        self.Vx = nn.Linear(self.args.max_seq_length, 1, bias=False)

        self.gru_mix_agg = nn.GRU(self.num_directions * self.hidden_size, self.hidden_size, batch_first=True,
                                  bidirectional=True)

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

        # concat attention
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

        x_concat = self.inside_aggregation((concat_rep, hp))
        x_billiner = self.inside_aggregation((billiner_rep, hp))
        x_dot = self.inside_aggregation((dot_rep, hp))
        x_minus = self.inside_aggregation((minus_rep, hp))

        aggregation = self.mix_aggregation([x_concat, x_billiner, x_dot, x_minus])

        self.gru_mix_agg.flatten_parameters()
        aggregation_rep, _ = self.gru_mix_agg(aggregation)

        output = self.perdoct_layer([hp, aggregation_rep])

        return output

    def inside_aggregation(self, x):
        # 原文公式 8a-8e
        attn_rep, hp = x
        xj = torch.cat([attn_rep, hp], dim=2)
        gj = F.sigmoid(self.Wg(xj))
        xj_ = gj * xj
        self.gru_inter_agg.flatten_parameters()
        x, _ = self.gru_inter_agg(xj_)
        return x

    def mix_aggregation(self, x):
        # 原文公式 9a-9c
        xc, xb, xd, xm = x
        x_cat_temp = torch.cat([xc, xb, xd, xm], dim=2)
        # B, L, 4, num_direction * rnn_hidden
        x_cat = x_cat_temp.view([x_cat_temp.size(0), x_cat_temp.size(1), 4, xc.size(2)]).contiguous()
        # B, L, 4
        x_cat_ = torch.squeeze(self.Wx(x_cat))
        # (B, L, 4 + L, 4)-> B, 4, L with L*1 -> B, 4, 1
        x_cat_v = x_cat_ + self.vx
        cross_attn = F.softmax(self.Vx(x_cat_v.transpose(2, 1)))
        # B, L, 1, num_direction * rnn_hidden
        x = torch.einsum("blad,bac->bldc",
                         x_cat,
                         cross_attn).squeeze()
        # x = cross_attn.bmm(x_cat)
        return x

    def perdoct_layer(self, x):
        # 原文公式 11a-11c
        # 比原文少一个 vp向量
        hp, aggregation_rep = x
        sj = self.Vp(torch.tanh(self.Wp(hp))).transpose(2, 1)
        rp = F.softmax(sj, 2).bmm(hp)

        # 原文12a~12c公式
        sj = F.softmax(self.V(self.W1(aggregation_rep) + self.W2(rp)).transpose(2, 1), 2)
        rc = sj.bmm(aggregation_rep)

        # 归一化
        # encoder_output = F.sigmoid(self.prediction(rc))
        output = self.output(rc.squeeze())

        return output


# Bimpm
class bimpm(nn.Module):
    def __init__(self,
                 embedd=None,
                 num_perspective=4,
                 hidden_size=128,
                 dropout_rate=0.5, ):
        super(bimpm, self).__init__()

        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim
        self.hidden_size = hidden_size
        self.num_perspective = num_perspective

        self.context_LSTM = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Matching Layer
        for i in range(1, 9):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.num_perspective,
                                            self.hidden_size)))

        # Aggregation Layer
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.num_perspective * 8,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Prediction Layer
        self.pred_fc1 = nn.Linear(
            self.hidden_size * 4,
            self.hidden_size * 2)
        self.pred_fc2 = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=self.hidden_size * 2)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        p, h = inputs['text_a'], inputs['text_b']

        # [B, L, D]
        # [B, R, D]
        p = self.embedding(p)
        h = self.embedding(h)

        p = self.dropout(p)
        h = self.dropout(h)

        # Context Representation Layer
        # (batch, seq_len, hidden_size * 2)
        self.context_LSTM.flatten_parameters()
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p,
                                         self.hidden_size,
                                         dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h,
                                         self.hidden_size,
                                         dim=-1)
        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        #   -> (batch, seq_len, num_perspective)
        mv_p_full_fw = mp_matching_func(
            con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(
            con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(
            con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(
            con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching

        # (batch, seq_len1, seq_len2, num_perspective)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, num_perspective)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = mp_matching_attention(con_p_fw, con_h_fw)
        att_bw = mp_matching_attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # output:  -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # output:  -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1)
        # output:  -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(
            att_h_fw.sum(dim=2),
            att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(
            att_h_bw.sum(dim=2),
            att_bw.sum(dim=2, keepdim=True))
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1)
        # output:  -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(
            att_p_fw.sum(dim=1),
            att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(
            att_p_bw.sum(dim=1),
            att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

        # (batch, seq_len, num_perspective)
        mv_p_att_mean_fw = mp_matching_func(
            con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(
            con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(
            con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(
            con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, num_perspective)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # (batch, seq_len, num_perspective * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw],
            dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw],
            dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # Aggregation Layer
        # (batch, seq_len, num_perspective * 8) -> (2, batch, hidden_size)
        self.aggregation_LSTM.flatten_parameters()
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2)
        #   -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(
                -1, self.hidden_size * 2),
                agg_h_last.permute(1, 0, 2).contiguous().view(
                    -1, self.hidden_size * 2)],
            dim=1)
        x = self.dropout(x)

        # Prediction Layer
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x
    

# ESIM
class ESIM(nn.Module):
    def __init__(self,
                 args,
                 embedd=None,
                 num_layer=1,
                 rnn_type="lstm",
                 drop_rnn=False,
                 drop_rate=0.2,
                 concat_rnn=True,
                 padding_idx=0
                 ):
        super(ESIM, self).__init__()

        self.args = args
        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim
        self.hidden_size = args.hidden_size
        self.num_layer = num_layer
        self.rnn_type = rnn_type
        self.drop_rnn = drop_rnn
        self.drop_rate = drop_rate
        self.concat_rnn = concat_rnn
        self.padding_idx = padding_idx

        self.build()

    def build(self):
        rnn_mapping = {'lstm': nn.LSTM, 'gru': nn.GRU}
        rnn_size = self.hidden_size
        if self.concat_rnn:
            rnn_size /= self.num_layer

        self.rnn_encoder = StackedBRNN(
            input_size=self.embedding_dim,
            hidden_size=int(rnn_size/2),
            num_layers=self.num_layer,
            dropout_rate=self.drop_rate ,
            dropout_output=self.drop_rnn,
            rnn_type=rnn_mapping[self.rnn_type],
            concat_layers=self.concat_rnn
        )

        self.attention = BidirectionalAttention()

        self.projection = nn.Sequential(
            nn.Linear(
                4 * self.hidden_size,
                self.hidden_size),
            parse_activation("relu"))

        self.composition = StackedBRNN(
            self.hidden_size,
            int(rnn_size / 2),
            self.num_layer,
            dropout_rate=self.drop_rate,
            dropout_output=self.drop_rnn,
            rnn_type=rnn_mapping[self.rnn_type],
            concat_layers=self.concat_rnn)

        self.classification = nn.Sequential(
            nn.Dropout(
                p=self.drop_rate),
            nn.Linear(
                4 * self.hidden_size,
                self.hidden_size),
            nn.Tanh(),
            nn.Dropout(
                p=self.drop_rate)
        )
        self.out = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=self.hidden_size)

    def forward(self, inputs):
        """Forward."""

        # [B, L], [B, R]
        query, doc = inputs['text_a'], inputs['text_b']

        # [B, L]
        # [B, R]
        query_mask = (query == self.padding_idx)
        doc_mask = (doc == self.padding_idx)

        # [B, L, D]
        # [B, R, D]
        query = self.embedding(query)
        doc = self.embedding(doc)

        # [B, L, D]
        # [B, R, D]
        # query = self.rnn_dropout(query)
        # doc = self.rnn_dropout(doc)

        # [B, L, H]
        # [B, R, H]
        # query = self.rnn_encoder(query, query_mask)
        # doc = self.rnn_encoder(doc, doc_mask)
        query = self.rnn_encoder(query)
        doc = self.rnn_encoder(doc)

        # [B, L, H], [B, L, H]
        attended_query, attended_doc = self.attention(
            query, query_mask, doc, doc_mask)

        # [B, L, 4 * H]
        # [B, L, 4 * H]
        # Enhancement of local inference information
        enhanced_query = torch.cat([query,
                                    attended_query,
                                    query - attended_query,
                                    query * attended_query],
                                   dim=-1)
        enhanced_doc = torch.cat([doc,
                                  attended_doc,
                                  doc - attended_doc,
                                  doc * attended_doc],
                                 dim=-1)
        # [B, L, H]
        # [B, L, H]
        projected_query = self.projection(enhanced_query)
        projected_doc = self.projection(enhanced_doc)

        # [B, L, H]
        # [B, L, H]
        # query = self.composition(projected_query, query_mask)
        # doc = self.composition(projected_doc, doc_mask)
        query = self.composition(projected_query)
        doc = self.composition(projected_doc)

        # [B, L]
        # [B, R]
        reverse_query_mask = 1. - query_mask.float()
        reverse_doc_mask = 1. - doc_mask.float()

        # [B, H] pooling
        query_avg = torch.sum(query * reverse_query_mask.unsqueeze(2), dim=1) \
                    / (torch.sum(reverse_query_mask, dim=1, keepdim=True) + 1e-8)
        doc_avg = torch.sum(doc * reverse_doc_mask.unsqueeze(2), dim=1) \
                  / (torch.sum(reverse_doc_mask, dim=1, keepdim=True) + 1e-8)

        # [B, L, H]
        # [B, L, H]
        query = query.masked_fill(query_mask.unsqueeze(2), -1e7)
        doc = doc.masked_fill(doc_mask.unsqueeze(2), -1e7)

        # [B, H]
        # [B, H]
        query_max, _ = query.max(dim=1)
        doc_max, _ = doc.max(dim=1)

        # [B, 4 * H]
        v = torch.cat([query_avg, query_max, doc_avg, doc_max], dim=-1)

        # [B, H]
        hidden = self.classification(v)

        # [B, num_classes]
        out = self.out(hidden)

        return out


# DIIN
class DIIN(nn.Module):
    def __init__(self,
                 args,
                 embedd,
                 conv_kernel_size: tuple = (3, 3),
                 pool_kernel_size: tuple = (2, 2),
                 dropout_rate: float = 0.2,
                 first_scale_down_ratio: float = 0.3,
                 transition_scale_down_ratio: float = 0.5,
                 padding_indx: int = 0):
        super(DIIN, self).__init__()

        """
        Deep Interactive Inference Network (IIN) consists of five component:
        embedding layer, encoding layer, interaction layer, feature extraction layer, and output layer.
        paper link: https://arxiv.org/pdf/1709.04348.pdf
        """
        self.args = args

        if embedd is None:
            raise Exception("The embdding layer is None")
        self.embedding = embedd
        self.embedding_dim = embedd.embedding_dim
        self.padding_idx = args.padding_idx
        self.num_char_embedding = args.num_char_embedding
        self.char_embedding_dim = args.char_embedding_dim
        self.char_conv_filters = args.char_conv_filters
        self.char_conv_kernel_size = args.char_conv_kernel_size

        self.first_scale_down_ratio = first_scale_down_ratio
        self.transition_scale_down_ratio = transition_scale_down_ratio
        self.nb_dense_blocks = args.nb_dense_blocks
        self.layers_per_dense_block = args.layers_per_dense_block
        self.growth_rate = args.growth_rate
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.dropout_rate = dropout_rate
        self.padding_idx = padding_indx

        self.build()

    def build(self):
        if self.args.use_char:
            self.char_embedding = CharacterEmbedding(
                char_embedding_input_dim=self.num_char_embedding,
                char_embedding_output_dim=self.char_embedding_dim,
                char_conv_filters=self.char_conv_filters,
                char_conv_kernel_size=self.char_conv_kernel_size
            )
            all_embed_dim = self.embedding_dim + self.char_conv_filters + 1
        else:
            all_embed_dim = self.embedding_dim + 1

        self.exact_maching = Matching(matching_type='exact')

        # Encoding
        self.left_encoder = SemanticComposite(
            all_embed_dim, self.dropout_rate)
        self.right_encoder = SemanticComposite(
            all_embed_dim, self.dropout_rate)

        # Interaction 原文公式(7)
        self.matching = Matching(matching_type='mul')

        # Feature Extraction
        self.conv = nn.Conv2d(
            in_channels=all_embed_dim,
            out_channels=int(all_embed_dim * self.first_scale_down_ratio),
            kernel_size=1
        )
        self.dense_net = DenseNet(
            in_channels=int(all_embed_dim * self.first_scale_down_ratio),
            nb_dense_blocks=self.nb_dense_blocks,
            layers_per_dense_block=self.layers_per_dense_block,
            growth_rate=self.growth_rate,
            transition_scale_down_ratio=self.transition_scale_down_ratio,
            conv_kernel_size=self.conv_kernel_size,
            pool_kernel_size=self.pool_kernel_size
        )
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))

        # Output
        self.out_layer = quickly_output_layer(
            task="classify",
            num_classes=2,
            in_features=self.dense_net.out_channels)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        # shape = [B, L]   shape = [B, R]
        input_word_left, input_word_right = inputs['text_a'], inputs['text_b']
        mask_word_left = (input_word_left == self.padding_idx)
        mask_word_right = (input_word_right == self.padding_idx)

        # shape = [B, L, D1]   shape = [B, R, D1]
        embed_word_left = self.dropout(self.embedding(input_word_left))
        embed_word_right = self.dropout(self.embedding(input_word_right))

        # shape = [B, L, 1]  shape = [B, R, 1]
        exact_match_left, exact_match_right = self.exact_maching(
            input_word_left, input_word_right)
        exact_match_left = exact_match_left.masked_fill(mask_word_left, 0)
        exact_match_right = exact_match_right.masked_fill(mask_word_right, 0)
        exact_match_left = torch.unsqueeze(exact_match_left, dim=-1)
        exact_match_right = torch.unsqueeze(exact_match_right, dim=-1)

        if self.args.use_char:
            # shape = [B, L, C]    shape = [B, R, C]
            input_char_left, input_char_right = inputs['text_a_char'], inputs['text_b_char']
            # shape = [B, L, D2]  shape = [B, R, D2]
            embed_char_left = self.dropout(self.char_embedding(input_char_left))
            embed_char_right = self.dropout(self.char_embedding(input_char_right))
            # shape = [B, L, D]   shape = [B, R, D]
            embed_left = torch.cat(
                [embed_word_left, embed_char_left, exact_match_left], dim=-1)
            embed_right = torch.cat(
                [embed_word_right, embed_char_right, exact_match_right], dim=-1)
        else:
            embed_left = torch.cat(
                [embed_word_left, exact_match_left], dim=-1)
            embed_right = torch.cat(
                [embed_word_right, exact_match_right], dim=-1)

        encode_left = self.left_encoder(embed_left)
        encode_right = self.right_encoder(embed_right)

        # shape = [B, L, R, D]
        interaction = self.matching(encode_left, encode_right)

        interaction = self.conv(self.dropout(interaction.permute(0, 3, 1, 2)))
        interaction = self.dense_net(interaction)
        interaction = self.max_pooling(interaction).squeeze(dim=-1).squeeze(dim=-1)

        output = self.out_layer(interaction)
        return output

    def SemanticComposite(self, x):
        # 原文公式 (1)~(6)
        seq_length = x.shape[1]

        x_1 = x.unsqueeze(dim=2).repeat(1, 1, seq_length, 1)
        x_2 = x.unsqueeze(dim=1).repeat(1, seq_length, 1, 1)
        x_concat = torch.cat([x_1, x_2, x_1 * x_2], dim=-1)

        # Self-attention layer.
        x_concat = self.dropout(x_concat)
        attn_matrix = self.att_linear(x_concat).squeeze(dim=-1)
        attn_weight = torch.softmax(attn_matrix, dim=2)
        attn = torch.bmm(attn_weight, x)

        # Semantic composite fuse gate.
        x_attn_concat = self.dropout(torch.cat([x, attn], dim=-1))
        # x_attn_concat = torch.cat([x, attn], dim=-1)
        z = torch.tanh(self.z_gate(x_attn_concat))
        r = torch.sigmoid(self.r_gate(x_attn_concat))
        f = torch.sigmoid(self.f_gate(x_attn_concat))
        encoding = r * x + f * z

        return encoding