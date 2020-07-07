import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder


class PointerNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_classes,
                 n_vocab,
                 embedded_pretrain=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 rnn_type="GRU",
                 return_type="generic",
                 padding_idx=None,
                 ):
        super(PointerNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab
        self.num_classes = num_classes

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.return_type = return_type
        self.padding_idx = padding_idx

        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.input_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.rnn = RNNEncoder(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              embedder=self.embedder,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout)


