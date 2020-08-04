# coding: utf-8
import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors

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

    def load_embeddings_from_gensim_model(self, gensim_model_path, stoi, scale=0.05):
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

    def load_embedding_from_gensim_vec(self, gensim_vec_path, stoi, scale=0.05, save=True):
        print("load {} num word embedding".format(len(stoi)))
        if save and gensim_vec_path.split(".")[-1] == "npy":
            print("load npy word embedding")
            weight = torch.tensor(np.load(gensim_vec_path))

        else:
            weight = torch.zeros(self.num_embeddings, self.embedding_dim)
            # model = KeyedVectors.load_word2vec_format(gensim_vec_path, limit=500000)
            model = KeyedVectors.load_word2vec_format(gensim_vec_path)
            num_known = 0
            for word in stoi:
                # try https://cloud.tencent.com/developer/article/1081196
                if word in model.wv.index2word:

                    weight[stoi[word]] = torch.from_numpy(model.wv[word])
                    num_known += 1
                else:
                    nn.init.uniform_(weight[stoi[word]], -scale, scale)
            print("{} words have pretrained embeddings".format(num_known),
                  "(coverage: {:.3f})".format(num_known / self.num_embeddings))
            if save:
                gensim_npz_path = gensim_vec_path.split(".")[0] + ".npy"
                np.save(gensim_npz_path, weight.numpy())

        self.weight.data.copy_(weight)



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


class CharacterEmbedding(nn.Module):
    """
    Character embedding module.
    :param char_embedding_input_dim: The input dimension of character embedding layer.
    :param char_embedding_output_dim: The output dimension of character embedding layer.
    :param char_conv_filters: The filter size of character convolution layer.
    :param char_conv_kernel_size: The kernel size of character convolution layer.
    """

    def __init__(
        self,
        char_embedding_input_dim: int = 100,
        char_embedding_output_dim: int = 8,
        char_conv_filters: int = 100,
        char_conv_kernel_size: int = 5
    ):
        """Init."""
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_embeddings=char_embedding_input_dim,
            embedding_dim=char_embedding_output_dim
        )
        self.conv = nn.Conv1d(
            in_channels=char_embedding_output_dim,
            out_channels=char_conv_filters,
            kernel_size=char_conv_kernel_size
        )

    def forward(self, x):
        """Forward."""
        embed_x = self.char_embedding(x)

        batch_size, seq_len, word_len, embed_dim = embed_x.shape

        embed_x = embed_x.contiguous().view(-1, word_len, embed_dim)

        embed_x = self.conv(embed_x.transpose(1, 2))
        embed_x = torch.max(embed_x, dim=-1)[0]

        embed_x = embed_x.view(batch_size, seq_len, -1)
        return embed_x