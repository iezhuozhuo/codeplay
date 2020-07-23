# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

from source.models.deepmatch import ArcI, ArcII, MVLSTM, MatchPyramid, MwAN, bimpm,ESIM


def ARCIConfig(parser):
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ARCIModel(args, embedd):
    model = ArcI(
        embedd=embedd,
        left_filters=[32],
        left_kernel_sizes=[3],
        left_pool_sizes=[2],
        right_filters=[32],
        right_kernel_sizes=[3],
        right_pool_sizes=[2],
        dropout_rate=0.4
    )
    return model


def ARCIIConfig(parser):
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ARCIIModel(args, embedd):
    model = ArcII(
        embedd=embedd,
        left_length=args.max_seq_length,
        right_length=args.max_seq_length,
        kernel_1d_count=32,
        kernel_1d_size=3,
        kernel_2d_count=[32],
        kernel_2d_size=[(3, 3)],
        pool_2d_size=[(2, 2)],
        conv_activation_func='relu',
        dropout_rate=0.5
    )
    return model


def MVLSTMConfig(parser):
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def MVLSTMoel(args, embedd):
    model = MVLSTM(
        embedd=embedd,
        hidden_size=128,
        num_layers=2,
        top_k=10,
        mlp_num_layers=2,
        mlp_num_units=64,
        mlp_num_fan_out=64,
        activation_func='relu',
        dropout_rate=0.5,
        bidirectional=True)

    return model


def MatchPyramidConig():
    pass


def MatchPyramidModel():
    pass


def MatchSRNNModel():
    pass


def MatchSRNNConfig():
    pass


def MwANModel(args, embedd):
    model = MwAN(
        args=args,
        embedd=embedd,
        num_layers=1,
        activation_func='relu',
        dropout_rate=0.5,
    )
    return model


def MwANConfig(parser):
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def BiMPMModule(args, embedd):
    model = bimpm(
        embedd=embedd,
        num_perspective=args.num_perspective,
        hidden_size=args.hidden_size,
        dropout_rate=0.5
    )
    return model


def BiMPMConfig(parser):
    parser.add_argument("--num_perspective", default=4, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ESIMModel(args, embedd):
    model = ESIM(
        args=args,
        embedd=embedd,
        num_layer=4,
        rnn_type="lstm",
        drop_rnn=True,
        drop_rate=0.2,
        concat_rnn=True,
        padding_idx=args.padding_idx
    )
    return model


def ESIMConfig(parser):
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args

