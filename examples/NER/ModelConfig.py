from transformers import WEIGHTS_NAME, BertConfig  # AlbertConfig

from source.models.ner import BiRNN_CRF, BiRNN_CRFOptimized


def RNNCRFModel(args, embedded_pretrain=None, vocab_size=0, padding_idx=None):
    if getattr(args, "optimized"):
        model = BiRNN_CRFOptimized(args=args,
                                   vocab_size=vocab_size,
                                   tag_to_ix=args.label2id,
                                   embedded_pretrain=embedded_pretrain,
                                   padding_idx=padding_idx,
                                   dropout=0.5, )
    else:
        model = BiRNN_CRF(args=args,
                          vocab_size=vocab_size,
                          tag_to_ix=args.label2id,
                          embedded_pretrain=embedded_pretrain,
                          padding_idx=padding_idx,
                          dropout=0.5, )

    return model


def RNNCRFConfig(parser):
    parser.add_argument("--embedded_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--optimized", action="store_true", help="whether use the optimized crf")

    args, _ = parser.parse_known_args()
    return args


def BertModelConfig(parser):
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument('--markup', default='bios', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--crf_learning_rate", type=float, default=2.5e-3)
    args, _ = parser.parse_known_args()
    args.bert = True
    return args, BertConfig
