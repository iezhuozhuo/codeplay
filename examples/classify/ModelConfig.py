import os

from source.models.classify import TextCNN, TextRNN, DPCNN, TransformerClassifier, FastText


def FastTextModel(args, embedded_pretrain, n_vocab=0, padding_idx=None, n_gram_vocab=250499, dropout=0.5):
    model = FastText(embedded_size=args.embedded_size,
                     hidden_size=args.hidden_size,
                     num_classes=args.num_classes,
                     n_vocab=n_vocab,
                     n_gram_vocab=n_gram_vocab,
                     dropout=dropout,
                     embedded_pretrain=embedded_pretrain,
                     padding_idx=padding_idx)
    return model


def TransformerClassifierModel(args, embedded_pretrain, n_vocab=0, padding_idx=None, n_position=200):
    model = TransformerClassifier(args=args,
                                  n_vocab=n_vocab,
                                  embedded_pretrain=embedded_pretrain,
                                  padding_idx=padding_idx,
                                  n_position=n_position)
    return model


def DPCNNModel(args, embedded_pretrain, n_vocab=0, padding_idx=None):
    model = DPCNN(num_filters=args.num_filters,
                  n_vocab=n_vocab,
                  max_length=args.max_seq_length,
                  num_classes=args.num_classes,
                  embedded_size=args.embedded_size,
                  embedded_pretrain=embedded_pretrain,
                  padding_idx=padding_idx)
    return model


def TextRNNModel(args, embedded_pretrain, n_vocab=0, padding_idx=None):
    return_type = "pool" if args.pool_arg else "generic"
    pooling_size = args.pool_arg if args.pool_arg else None

    model = TextRNN(input_size=args.embedded_size,
                    hidden_size=args.hidden_size,
                    num_classes=args.num_classes,
                    n_vocab=n_vocab,
                    embedded_pretrain=embedded_pretrain,
                    num_layers=args.num_layers,
                    bidirectional=True,
                    dropout=0.5,
                    padding_idx=padding_idx,
                    pooling_size=pooling_size,
                    return_type=return_type
                    )

    return model


def TextCNNModel(args, embedded_pretrain, n_vocab=0, padding_idx=None):
    model = TextCNN(num_filters=args.num_filters,
                    embedded_size=args.embedded_size,
                    dropout=0.5,
                    num_classes=args.num_classes,
                    n_vocab=n_vocab,
                    filter_sizes=args.filter_sizes,
                    embedded_pretrain=embedded_pretrain,
                    padding_idx=padding_idx,
                    )
    return model


def DPCNNConfig(parser):
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument(
        "--label_file",
        default="class.txt",
        type=str,
        help="The input label file.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./output/vocab.json",
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument(
        "--embed_file",
        default="D:\\BaiduNetdiskDownload\\sgns.sogou.char",
        # default=None,
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument("--embedded_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def TextRNNConfig(parser):
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--pool_arg", type=int, default=None)
    parser.add_argument(
        "--label_file",
        default="class.txt",
        type=str,
        help="The input label file.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./output/vocab.json",
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument(
        "--embed_file",
        default="D:\\BaiduNetdiskDownload\\sgns.sogou.char",
        # default=None,
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument("--embedded_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def TextCNNConfig(parser):
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument(
        "--label_file",
        default="class.txt",
        type=str,
        help="The input label file.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./output/vocab.json",
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument(
        "--embed_file",
        default="D:\\BaiduNetdiskDownload\\sgns.sogou.char",
        # default=None,
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument("--embedded_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def TransformerClassifierConfig(parser):
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=6)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-6)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.55)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.55)

    parser.add_argument(
        "--label_file",
        default="class.txt",
        type=str,
        help="The input label file.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./output/vocab.json",
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument(
        "--embed_file",
        default="D:\\BaiduNetdiskDownload\\sgns.sogou.char",
        # default=None,
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument("--embedded_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def FastTextConfig(parser):
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_gram_vocab', type=int, default=250499)

    parser.add_argument(
        "--label_file",
        default="class.txt",
        type=str,
        help="The input label file.",
    )
    parser.add_argument(
        "--vocab_path",
        default="./output/vocab.json",
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument(
        "--embed_file",
        default="D:\\BaiduNetdiskDownload\\sgns.sogou.char",
        # default=None,
        type=str,
        help="The input pretrain embedded file.",
    )
    parser.add_argument("--embedded_size", type=int, default=300)
    parser.add_argument("--num_classes", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args
