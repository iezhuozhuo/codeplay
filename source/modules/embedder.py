# coding: utf-8
import pandas as pd
import numpy as np
import gensim

import torch
import torch.nn as nn


class Embedder(nn.Embedding):
    """ Generic Embedder """
    def load_embeddings(self, embeds, scale=0.05):
        """
        load_embeddings
        """
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))

    def load_embeddingsfor_gensim_vec(self, gensim_model_path, stoi, scale=0.05):
        weight = torch.zeros(self.num_embeddings, self.embedding_dim)
        model = gensim.models.Word2Vec.load(gensim_model_path)
        # word_vec_dict = pd.read_csv(file, encoding="utf-8")
        num_known = 0
        for word in stoi:
            if word in model.wv.index2word:
                weight[stoi[word]] = torch.from_numpy(model.wv[word])
                num_known += 1
            else:
                nn.init.uniform_(weight[stoi[word]], -scale, scale)

        self.weight.data.copy_(weight)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # Tod: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
