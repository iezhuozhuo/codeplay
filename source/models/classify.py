# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.resnet import ResnetBlock
from source.modules.embedder import PositionalEncoding
from source.modules.encoders.BertFashionEncoder import BertFashionLayer


class TextCNN(nn.Module):
    def __init__(self,
                 num_filters, embedded_size, dropout, num_classes,
                 n_vocab, filter_sizes=(2, 3, 4), embedded_pretrain=None, padding_idx=None):
        super().__init__()

        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.embedded_size = embedded_size
        self.num_classes = num_classes
        self.n_vocab = n_vocab
        self.padding_idx = padding_idx if padding_idx else self.n_vocab - 1

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
        out = self.embedder(x[0])
        # out = self.embedder(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        pred = self.fc(out)
        return pred


class TextRNN(nn.Module):
    """
    return_type: generic-rnn  pool-rcnn
    """
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
                 pooling_size=None):
        super().__init__()

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
        self.pooling_size = pooling_size
        self.fc_size = input_size+hidden_size if self.return_type == "pool" else self.hidden_size

        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.input_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        # get different rnn type using rnn_type  using new GRU or LSTM
        self.rnn_encoder = RNNEncoder(input_size=self.input_size,
                                      hidden_size=self.hidden_size,
                                      embedder=self.embedder,
                                      num_layers=self.num_layers,
                                      bidirectional=self.bidirectional,
                                      dropout=self.dropout)

        if self.return_type == "pool":
            assert self.pooling_size is not None, "if you choose rcnn, please input pooling_size"
            self.maxpooling = nn.MaxPool1d(self.pooling_size)

        self.fc = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, inputs, hidden=None):
        outputs, last_hidden = self.rnn_encoder(inputs[0], hidden)

        if self.return_type == "generic":
            pred = self.fc(last_hidden[-1])
        elif self.return_type == "pool":
            embed = self.rnn_encoder.embedder(inputs[0])
            out = torch.cat((embed, outputs), 2)
            out = F.relu(out)
            out = out.permute(0, 2, 1)
            pool_out = self.maxpooling(out).squeeze()
            pred = self.fc(pool_out)

        return pred


class DPCNN(nn.Module):
    def __init__(self,
                 num_filters,
                 n_vocab,
                 max_length,
                 num_classes,
                 embedded_size,
                 embedded_pretrain=None,
                 padding_idx=None):
        super(DPCNN, self).__init__()

        self.num_filters = num_filters
        self.n_vocab = n_vocab
        self.max_length = max_length
        self.num_classes = num_classes
        self.embedded_size = embedded_size
        self.padding_idx = padding_idx if padding_idx else self.n_vocab - 1

        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.embedded_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        # tri-gram + bn + pre-activate + dropout
        self.region_embedding = nn.Sequential(
            nn.Conv1d(self.embedded_size, self.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.num_filters),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 等长卷积+pre-activate
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.num_filters),
            nn.ReLU(),
            nn.Conv1d(self.num_filters, self.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.num_filters),
            nn.ReLU(),
            nn.Conv1d(self.num_filters, self.num_filters,
                      kernel_size=3, padding=1),
        )

        resnet_block_list = []
        while self.max_length > 2:
            resnet_block_list.append(ResnetBlock(self.num_filters))
            self.max_length = self.max_length // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(self.num_filters * self.max_length, self.num_classes),
            nn.BatchNorm1d(self.num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.num_classes, self.num_classes)
        )

    def forward(self, inputs):
        x = self.embedder(inputs[0])
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(batch_size, -1)
        out = self.fc(x)
        return out


class TransformerClassifier(nn.Module):
    def __init__(self,
                 args,
                 n_vocab,
                 embedded_pretrain=None,
                 padding_idx=None,
                 n_position=200):
        super(TransformerClassifier, self).__init__()

        self.args = args
        self.n_vocab = n_vocab
        self.num_classes = args.num_classes
        self.embedded_size = args.embedded_size
        self.num_layers = args.num_layers
        self.n_position = n_position

        self.padding_idx = padding_idx
        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.embedded_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.embedded_position = PositionalEncoding(self.embedded_size, n_position=self.n_position)

        self.layer_stack = nn.ModuleList([
            BertFashionLayer(self.args)
            for _ in range(self.num_layers)])

        self.fc = nn.Linear(self.args.hidden_size * self.args.max_seq_length, self.num_classes)

    def forward(self,
                inputs,
                attention_mask=None,
                output_attentions=False):
        outputs = self.embedded_position(self.embedder(inputs[0]))
        # 第一层hidden_size != embeded_size 就会麻烦
        for enc_layer in self.layer_stack:
            outputs, weights = enc_layer(outputs, attention_mask=attention_mask, output_attentions=output_attentions)

        outputs = outputs.view(outputs.size(0), -1)
        # outputs = torch.mean(outputs, 1)
        pred = self.fc(outputs)
        return pred


class FastText(nn.Module):
    def __init__(self,
                 embedded_size,
                 hidden_size,
                 num_classes,
                 n_vocab,
                 n_gram_vocab,
                 dropout=0.5,
                 embedded_pretrain=None,
                 padding_idx=None):
        super(FastText, self).__init__()

        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_vocab = n_vocab
        self.n_gram_vocab = n_gram_vocab
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.embedded_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.embedder_ngram2 = nn.Embedding(self.n_gram_vocab, self.embedded_size)
        self.embedder_ngram3 = nn.Embedding(self.n_gram_vocab, self.embedded_size)

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.embedded_size * 3, self.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        out_word = self.embedder(x[0])
        out_bigram = self.embedder_ngram2(x[2])
        out_trigram = self.embedder_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


