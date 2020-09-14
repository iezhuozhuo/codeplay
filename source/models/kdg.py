# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!
# This py is for knowledge-based dialogue generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import LSTMEncoder, GRUEncoder, StackedBRNN


# TODO MemGM
class MemGM(nn.Module):
    def __int__(self,
                args,
                src_vocab_size=50000,
                tgt_vocab_size=50000,
                src_prembed=None,
                tgt_prembed=None,
                padding_idx=0):
        super(MemGM, self).__int__()
        self.args = args
        self.padding_idx = padding_idx

        enc_embedding = Embedder(src_vocab_size, args.embedding_size, padding_idx)
        dec_embedding = Embedder(tgt_vocab_size, args.embedding_size, padding_idx)
        if self.args.embedding_type == 1:
            # 1:[all spread: 1)src_enc 2)tgt_enc 3)tgt_dec] 2:[share enc: 1)src_enc,tgt_enc 2)tgt_dec]
            # 3:[share tgt: 1)src_enc 2)tgt_enc,tgt_dec] 4:[all share: 1)src_enc,tgt_enc,tgt_dec (not implement)]
            src_enc_embedding = Embedder(src_vocab_size, args.embedding_size, padding_idx)
            self.c_encoder = self.create_encoder(src_enc_embedding)
        else:
            self.c_encoder = self.create_encoder(enc_embedding)

        if self.args.embedding_type == 1:
            self.x_encoder = self.create_encoder(enc_embedding)
        elif self.args.embedding_type in [3]:
            self.x_encoder = self.create_encoder(dec_embedding)

    def create_encoder(self, src_enc_embedding):
        if self.args.rnn_type == "lstm":
            return LSTMEncoder(
                input_size=self.args.embedding_size,
                hidden_size=self.args.hidden_size,
                rnn_hidden_size=None,
                embedder=src_enc_embedding,
                num_layers=self.args.num_layers,
                bidirectional=True,
                dropout=self.args.dropout,
                output_type="seq2seq"
            )
        else:
            return GRUEncoder(input_size=self.args.embedding_size,
                              hidden_size=self.args.hidden_size,
                              rnn_hidden_size=None,
                              embedder=src_enc_embedding,
                              num_layers=self.args.num_layers,
                              bidirectional=True,
                              dropout=self.args.dropout,
                              output_type="seq2seq"
                              )
