# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

from source.models.deepmatch import ArcI, ArcII, MVLSTM, MatchPyramid, MwAN, bimpm, ESIM, DIIN


def ARCIConfig(parser):
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--max_utter_len", default=50, type=int)
    parser.add_argument("--max_utter_num", default=10, type=int)
    parser.add_argument("--max_response_len", default=50, type=int)
    parser.add_argument("--gru_units", default=200, type=int)
    parser.add_argument("--maxWordLength", default=18, type=int)
    parser.add_argument("--att_dim", default=200, type=int)
    parser.add_argument("--max_to_keep", default=1, type=int)
    parser.add_argument("--cnn_filters", default=50, type=int)
    parser.add_argument(
        "--embedded_vector_file",
        default='/home/administrator4/ZDY/dataset/Ubuntu_Corpus_V1/glove_42B_300d_vec_plus_word2vec_100.txt',
        type=str)
    parser.add_argument(
        "--vocab_file",
        default='/home/administrator4/ZDY/dataset/Ubuntu_Corpus_V1/vocab.txt',
        type=str)
    parser.add_argument(
        "--char_vocab_file",
        default='/home/administrator4/ZDY/dataset/Ubuntu_Corpus_V1/char_vocab.txt',
        type=str)
    parser.add_argument(
        "--response_file",
        default='/home/administrator4/ZDY/dataset/Ubuntu_Corpus_V1/responses.txt',
        type=str)
    parser.add_argument("--dropout_keep_prob", default=1.0, type=int)
    parser.add_argument("--f_hidden", default=25, type=int)
    parser.add_argument("--decay_step", default=781, type=int)
    args, _ = parser.parse_known_args()
    return args
'''
    "max_utter_len":50,
    "max_utter_num":10,
    "max_response_len":50, 
    "vocab_size":144952, 
    "emb_size":400, 
    "embedding_dim":400,
    "embbing_size":550,
    "gru_units":200, 
    "maxWordLength":18, 
    "att_dim":200,
    "learning_rate":0.001,
    "max_to_keep":1,
    "batch_size":67,
    "cnn_filters":50,
    "embedded_vector_file" : "./data/Ubuntu_Corpus_V1/glove_42B_300d_vec_plus_word2vec_100.txt",
    "vocab_file":"./data/Ubuntu_Corpus_V1/vocab.txt",
    "char_vocab_file":"./data/Ubuntu_Corpus_V1/char_vocab.txt",
    "response_file":"./data/Ubuntu_Corpus_V1/responses.txt",
    "train_file":"./data/Ubuntu_Corpus_V1/train.txt",
    "valid_file":"./data/Ubuntu_Corpus_V1/valid.txt",
    "test_file":"./data/Ubuntu_Corpus_V1/test.txt",
    "save_path":"/home/gong/zz/ChatBot/zz/zz/最终版本/output/ubuntu/ubuntu_cnn/",
    "num_epochs":10,
    "decay_step":781,
    "dropout_keep_prob":1.0,
    "init_model":None
    '''
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


def BiMPMModule(args, embedd, char_embedd):
    model = bimpm(
        embedd=embedd,
        char_embedd=char_embedd,
        num_perspective=args.num_perspective,
        hidden_size=args.hidden_size,
        dropout_rate=0.5
    )
    return model

def BiMPMConfig(parser):
    parser.add_argument("--num_perspective", default=20, type=int)
    parser.add_argument("--max_char_seq_length", default=16, type=int)
    parser.add_argument('--char_embedding_dim', default=20, type=int)
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
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--use_char", action="store_true")
    parser.add_argument('--char_embedding_dim', default=8, type=int)
    parser.add_argument('--char_conv_filters', default=100, type=int)
    parser.add_argument('--char_conv_kernel_size', default=5, type=int)
    parser.add_argument('--nb_dense_blocks', default=3, type=int)
    parser.add_argument('--layers_per_dense_block', default=8, type=int)
    parser.add_argument('--growth_rate', default=20, type=int)
    parser.add_argument("--max_char_seq_length", default=5, type=int)
    args, _ = parser.parse_known_args()
    return args
