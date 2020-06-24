# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models import base_model
from source.modules.embedder import Embedder


class TextCNN(nn.Module):
    def __init__(self,
                 num_filters, embedded_size, dropout, num_classes,
                 n_vocab, filter_sizes=(2,3,4), embedded_pretrain=None, padding_idx=None):

        super().__init__()

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.embedded_size = embedded_size
        self.num_classes = num_classes
        self.n_vocab = n_vocab
        self.padding_idx = padding_idx if padding_idx else self.n_vocab-1

        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.embedded_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embedded_size)) for k in self.filter_sizes])
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # out = self.embedder(x[0])
        out = self.embedder(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        pred = self.fc(out)
        return pred