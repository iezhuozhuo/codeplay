# coding: utf-8

import torch
import torch.nn as nn

from source.models.crf import CRF
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder


class BiRNN_CRF(nn.Module):
    def __init__(self,
                 args,
                 vocab_size,
                 tag_to_ix,
                 embedded_pretrain=None,
                 padding_idx=None,
                 dropout=0.5,
                 ):
        super(BiRNN_CRF, self).__init__()

        self.args = args
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(self.tag_to_ix)

        self.embedded_size = args.embedded_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = dropout
        self.bidirectional = True

        self.padding_idx = padding_idx
        self.embedder = Embedder(num_embeddings=self.vocab_size,
                                 embedding_dim=self.embedded_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.rnn_encoder = RNNEncoder(input_size=self.embedded_size,
                                      hidden_size=self.hidden_size,
                                      embedder=self.embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        self.hidden2tag = nn.Linear(self.hidden_size, self.tagset_size)

        self.crf = CRF(tagset_size=self.tagset_size,
                       tag_dictionary=self.tag_to_ix,
                       device=self.args.device,
                       )

    def forward(self, inputs, labels=None, hidden=None):
        outputs, last_hidden = self.rnn_encoder(inputs[0], hidden)
        logits = self.hidden2tag(outputs)
        outputs = (logits,)

        if labels is not None:
            loss = self.crf.calculate_loss(features=logits, tag_list=labels, lengths=inputs[1])
            outputs = (loss,) + outputs

        return outputs