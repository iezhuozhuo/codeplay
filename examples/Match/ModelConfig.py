# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!
from source.models.deepmatch import MVLSTM, BiMPM_Model

def ARCIConfig():
    pass


def ARCIModel():
    pass


def ARCIIConfig():
    pass


def ARCIIModel():
    pass


def MVLSTMConfig(parser):
    parser.add_argument(
        "--aug", action="store_true", help="data augment"
    )
    # parser.add_argument("--max_seq_length", type=int, default=25)
    # parser.add_argument("--max_right_seq_length", type=int, default=25)
    # parser.add_argument("--embedded_size", type=int, default=128)
    # parser.add_argument("--hidden_size", type=int, default=32)
    # parser.add_argument("--out_size", type=int, default=2)
    # parser.add_argument( "--train_file", type=str, default="atec_nlp_sim_train_all.csv")
    args, _ = parser.parse_known_args()
    return args


def MVLSTModel(args, embedded_pretrain, vocab_size, padding_idx):
    model = MVLSTM(embedded_pretrain)
    return model


def MatchPyramidConig():
    pass


def MatchPyramidModel():
    pass

def BiMPM(args, embedded_pretrain, vocab_size, padding_idx):
    model = BiMPM_Model(embedded_pretrain)
    return model