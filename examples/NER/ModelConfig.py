from source.models.ner import BiRNN_CRF


def RNNCRFModel(args, embedded_pretrain=None, vocab_size=0, padding_idx=None):
    model = BiRNN_CRF(args=args,
                      vocab_size=vocab_size,
                      tag_to_ix=args.label2id,
                      embedded_pretrain=embedded_pretrain,
                      padding_idx=padding_idx,
                      dropout=0.5, )

    return model


def RNNCRFConfig(parser):
    parser.add_argument("--embedded_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)

    args, _ = parser.parse_known_args()
    return args
