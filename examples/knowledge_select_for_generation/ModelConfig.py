# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

from source.models.deepmatch import ArcI, ArcII, MVLSTM, MatchPyramid, MwAN, bimpm, ESIM, DIIN
from source.models.knowledge_seq2seq import KnowledgeSeq2Seq

def ARCIConfig(parser):
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--with_label", default=False)
    data_arg.add_argument("--embed_file", type=str, default=None)

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", default=True)
    net_arg.add_argument("--max_vocab_size", type=int, default=30000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                        choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", default=True)
    net_arg.add_argument("--with_bridge", default=True)
    net_arg.add_argument("--tie_embedding", default=True)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.00005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--num_epochs", type=int, default=20)
    train_arg.add_argument("--pretrain_epoch", type=int, default=5)
    train_arg.add_argument("--lr_decay", type=float, default=None)
    train_arg.add_argument("--use_embed", default=True)
    train_arg.add_argument("--use_bow", default=True)
    train_arg.add_argument("--use_dssm", default=False)
    train_arg.add_argument("--use_pg",  default=False)
    train_arg.add_argument("--use_gs", default=False)
    train_arg.add_argument("--use_kd", default=False)
    train_arg.add_argument("--weight_control",  default=False)
    train_arg.add_argument("--decode_concat",  default=False)
    train_arg.add_argument("--use_posterior", default=True)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--max_dec_len", type=int, default=30)
    gen_arg.add_argument("--ignore_unk", default=True)
    gen_arg.add_argument("--length_average", default=True)
    gen_arg.add_argument("--gen_file", type=str, default="./test.result")
    gen_arg.add_argument("--gold_score_file", type=str, default="./gold.scores")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=-1)
    misc_arg.add_argument("--log_steps", type=int, default=100)
    misc_arg.add_argument("--valid_steps", type=int, default=200)
    misc_arg.add_argument("--batch_size", type=int, default=128)
    misc_arg.add_argument("--ckpt", type=str)                                       # 从保存的模型的epoch开始训练（网络要一致）
    #misc_arg.add_argument("--ckpt", type=str, default="models/best.model")
    misc_arg.add_argument("--check", action="store_true")                           # 将文件保存在tmp而不是models
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--interact", action="store_true")
    #misc_arg.add_argument("--interact", type=str2bool, default=True)
    args, _ = parser.parse_known_args()
    return args


def ARCIModel(args, embedd, field):
    model = KnowledgeSeq2Seq(src_vocab_size=field['text'].vocab_size,
                             tgt_vocab_size=field['text'].vocab_size,
                             embed_size=args.embed_size, hidden_size=args.hidden_size,
                             padding_idx=args.padding_idx,
                             num_layers=args.num_layers, bidirectional=args.bidirectional,
                             attn_mode=args.attn, with_bridge=args.with_bridge,
                             tie_embedding=args.tie_embedding, dropout=args.dropout,
                             use_gpu=True,
                             use_bow=args.use_bow, use_dssm=args.use_dssm,
                             use_pg=args.use_pg, use_gs=args.use_gs,
                             pretrain_epoch=args.pretrain_epoch,
                             use_posterior=args.use_posterior,
                             weight_control=args.weight_control,
                             concat=args.decode_concat)
    return model

