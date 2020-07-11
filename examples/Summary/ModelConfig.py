# encoding utf-8
from source.models.pointer_network import PointerNet


def configs(parser):
    parser.add_argument("--max_enc_seq_length", type=int, default=128)
    parser.add_argument("--max_dec_seq_length", type=int, default=64)
    parser.add_argument("--max_oov_len", type=int, default=64)
    parser.add_argument("--embedded_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--min_dec_steps", type=int, default=35)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--pointer_gen", action="store_true", help="whether use the pointer_gen")

    parser.add_argument(
        "--article_file",
        default="train_text.txt",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--summary_file",
        default="train_label.txt",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    args, _ = parser.parse_known_args()
    return args


def PointNetworkModel(args, embedded_pretrain=None, vocab_size=0, padding_idx=None):
    model = PointerNet(input_size=args.embedded_size,
                       hidden_size=args.hidden_size,
                       n_vocab=vocab_size,
                       embedded_pretrain=embedded_pretrain,
                       num_layers=1,
                       bidirectional=True,
                       dropout=0.0,
                       padding_idx=padding_idx)
    return model
