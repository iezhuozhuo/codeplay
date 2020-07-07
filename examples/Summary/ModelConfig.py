# encoding utf-8


def configs(parser):
    parser.add_argument("--max_enc_seq_length", type=int, default=128)
    parser.add_argument("--max_dec_seq_length", type=int, default=64)
    parser.add_argument("--max_oov_len", type=int, default=64)
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