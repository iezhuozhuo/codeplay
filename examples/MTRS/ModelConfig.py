import os
from source.models.ResponseSelect import SMN


def SMNModel(args, embedded_pretrain=None):
    model = SMN(args=args, embedded_pretrain=embedded_pretrain)
    return model


def SMNConfig(parser):
    parser.add_argument("--num_embeddings", type=int, default=434511, help="--setting for nn.Embedding")
    parser.add_argument("--input_size", type=int, default=50, help="--set max length of each sequence")
    parser.add_argument("--embedding_size", type=int, default=200, help="--setting for nn.Embedding")
    parser.add_argument("--hidden_size", type=int, default=200, help="--GRU hidden size")

    parser.add_argument("--out_channels", type=int, default=8, help="--number of output channels of conv2d")
    parser.add_argument("--kernel_size", type=tuple, default=(3, 3), help="--set kernel_size of conv2d")
    parser.add_argument("--stride", type=tuple, default=(3, 3), help="--set stride of maxpooling2d")
    parser.add_argument("--inter_size", type=int, default=50,
                        help="--set output representation size of Matching Module")
    parser.add_argument("--hidden_size_ma", type=int, default=50, help="--set GRU hidden size for MatchAcc Module")
    parser.add_argument("--q", type=int, default=50, help="--set q")
    parser.add_argument("--device",default="cuda",type=str,)

    # 限定最大对话轮次
    parser.add_argument("--max_utter_num",default=10,type=int,)
    parser.add_argument("--fusion_type", type=str, default='dynamic', help="--set fusion type")

    args, _ = parser.parse_known_args()
    return args


# def IMNModel(args):
#     model = IMN(args=args)
#     return model
#
#
# def IMNConfig(parser):
#     args, _ = parser.parse_known_args()
#     return args