# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

from source.models.deepmatch import ArcI, ArcII, MVLSTM, MatchPyramid, MwAN, bimpm, ESIM, DIIN


def ARCIConfig(parser):
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--mlp_num_layers", default=2, type=int)
    parser.add_argument("--mlp_num_units", default=128, type=int)
    parser.add_argument("--mlp_num_fan_out", default=64, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ARCIModel(args, embedd):
    model = ArcI(
        args=args,
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
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def ARCIIModel(args, embedd):
    model = ArcII(
        args=args,
        embedd=embedd,
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
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--mlp_num_layers", default=2, type=int)
    parser.add_argument("--mlp_num_units", default=128, type=int)
    parser.add_argument("--mlp_num_fan_out", default=64, type=int)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def MVLSTMoel(args, embedd):
    model = MVLSTM(
        args=args,
        embedd=embedd,
        hidden_size=args.hidden_size,
        num_layers=2,
        top_k=args.top_k,
        activation_func='relu',
        dropout_rate=0.5,
        bidirectional=True)

    return model


def MatchPyramidConig(parser):
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


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
    parser.add_argument("--num_class", default=2, type=int)
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
    parser.add_argument("--num_class", default=2, type=int)
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
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def DIINModel(args, embedd):
    model = DIIN(
        args=args,
        embedd=embedd,
        conv_kernel_size=(3, 3),
        pool_kernel_size=(2, 2),
        dropout_rate=0.2,
        first_scale_down_ratio=0.3,
        transition_scale_down_ratio=0.5,
        padding_indx=args.padding_idx
    )
    return model


def DIINConfig(parser):
    parser.add_argument("--num_class", default=3, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--use_char", action="store_true")
    parser.add_argument('--char_embedding_dim', default=8, type=int)
    parser.add_argument('--char_conv_filters', default=128, type=int)
    parser.add_argument('--char_conv_kernel_size', default=2, type=int)
    parser.add_argument('--nb_dense_blocks', default=3, type=int)
    parser.add_argument('--layers_per_dense_block', default=8, type=int)
    parser.add_argument('--growth_rate', default=20, type=int)
    parser.add_argument("--max_char_seq_length", default=16, type=int)
    args, _ = parser.parse_known_args()
    return args
