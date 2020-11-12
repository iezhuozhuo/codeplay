import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import GRUEncoder
from source.modules.encoders.cnn_encoder import cnn_output_size
import numpy as np


class SMN(nn.Module):
    '''
    Refer to paper for details.
    Title:Sequential Matching Network:A New Architecture for Multi-turn
Response Selection in Retrieval-Based Chatbots
    Authors:Yu Wu†, Wei Wu‡, Chen Xing♦, Zhoujun Li†∗, Ming Zhou
    Tensorflow version : https://github.com/MarkWuNLP/MultiTurnResponseSelection
    '''
    def __init__(self,
                 args,
                 embedded_pretrain=None,
                 padding_idx=0) -> None:
        super().__init__()

        self.args = args
        self.padding_idx = padding_idx

        self.embedder = Embedder(num_embeddings=self.args.num_embeddings,
                                 embedding_dim=self.args.embedding_size, padding_idx=self.padding_idx)
        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.gru_utterance = GRUEncoder(
            input_size=self.args.embedding_size,
            hidden_size=self.args.hidden_size,
            rnn_hidden_size=self.args.hidden_size,
            embedder=self.embedder,
            bidirectional=False,
            output_type="general",
        )

        self.gru_response = GRUEncoder(
            input_size=self.args.embedding_size,
            hidden_size=self.args.hidden_size,
            rnn_hidden_size=self.args.hidden_size,
            embedder=self.embedder,
            bidirectional=False,
            output_type="general",
        )

        self.A = nn.Parameter(
            torch.randn(size=(self.args.hidden_size, self.args.hidden_size), requires_grad=True)
        )

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=self.args.out_channels,
            kernel_size=self.args.kernel_size
        )
        self.act = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=self.args.kernel_size, stride=self.args.stride)

        self.output_size = cnn_output_size(
            input_size=cnn_output_size(input_size=(self.args.input_size, self.args.input_size),
                                       filter_size=self.args.kernel_size,
                                       stride=(1, 1), padding=(0, 0)),
            filter_size=self.args.kernel_size, stride=self.args.stride, padding=(0, 0)
        )

        self.linear = nn.Linear(
            self.args.out_channels * self.output_size[0] * self.output_size[1],
            self.args.inter_size
        )

        self.gru = GRUEncoder(
            input_size=self.args.inter_size,
            hidden_size=self.args.hidden_size_ma,
            rnn_hidden_size=self.args.hidden_size_ma,
            bidirectional=False,
            output_type="general",
        )

        # no extra parameter for SMN-last
        if self.args.fusion_type == 'last':
            pass

        # parameters for SMN-static (w)
        if self.args.fusion_type == 'static':
            self.weight_static = nn.Parameter(torch.randn(size=(1, self.args.max_utter_num), requires_grad=True))

        # parameters for SMN-dynamic (W1,W2,ts)
        if self.args.fusion_type == 'dynamic':
            self.linear1 = nn.Linear(self.args.hidden_size * 2, self.args.q, bias=True)
            self.linear2 = nn.Linear(self.args.hidden_size_ma * 2, self.args.q, bias=False)
            self.ts = nn.Parameter(torch.randn(size=(self.args.q, 1)), requires_grad=True)

        self.lin_output = nn.Linear(self.args.hidden_size_ma * 2, 2, bias=True)

        self._init_para()

    def forward(self,
                utterance,
                len_utterance,
                num_utterance,
                response,
                len_response):

        match, h_u = self.urmatch(utterance,len_utterance,response,len_response)
        score = self.matchacc(num_utterance,match,h_u)
        return score

    def urmatch(self, utterance,len_utterance,response,len_response):
        utterance_embed = self.embedding(utterance)
        all_utterance_embeddings = utterance_embed.permute(1, 0, 2, 3)
        all_utterance_length = len_utterance.permute(1, 0)
        response_embed = self.embedding(response)
        response_embed_gru, _ = self.gru_response((response_embed, len_response))
        response_embeddings = response_embed.permute(0, 2, 1)
        response_embed_gru = response_embed_gru.permute(0, 2, 1)

        matching_vectors, h_u = [], []
        for utterance_embeddings, utterance_length in zip(all_utterance_embeddings, all_utterance_length):
            m1 = torch.matmul(utterance_embeddings, response_embeddings)

            # utterance_embed_gru, last_hidden_u = self.gru_utterance((utterance_embeddings,utterance_length))
            utterance_embed_gru, last_hidden_u = self.gru_utterance(utterance_embeddings)

            m2 = torch.einsum('aij,jk->aik', utterance_embed_gru, self.A)
            m2 = torch.matmul(m2, response_embed_gru)

            m = torch.stack([m1, m2], dim=1)

            conv_layer = self.conv(m)
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.pooling(conv_layer)
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)

            matching_vector = self.linear(pooling_layer)
            matching_vector = torch.tanh(matching_vector)
            matching_vectors.append(matching_vector)

            h_u.append(last_hidden_u.squeeze(0))

        match = torch.stack(matching_vectors, dim=1)
        # for dynamic fusion of h'
        h_u = torch.stack(h_u, dim=1)
        return match, h_u

    def matchacc(self, num_utterance,match,h_u):
        output, last_hidden = self.gru((match, num_utterance))
        # last:the last hidden state
        if self.args.fusion_type == 'last':
            L = last_hidden.squeeze(0)

        # static:weighted sum of all hidden states
        elif self.args.fusion_type == 'static':
            L = torch.matmul(self.weight_static, output).squeeze(1)

        # dynamic:attention-weighted hidden states
        elif self.args.fusion_type == 'dynamic':
            t = torch.tanh(self.linear1(h_u) + self.linear2(output))
            alpha = F.softmax(torch.matmul(t, self.ts), dim=1).squeeze(-1)
            L = torch.matmul(alpha.unsqueeze(1), output).squeeze(1)

        # deal with unbound variable
        else:
            L = 0
        g = self.lin_output(L)

        return g

    def _init_para(self):

        nn.init.xavier_uniform_(self.A)

        conv2d_weight = (param.data for name, param in self.conv.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            nn.init.kaiming_normal_(w)

        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            nn.init.xavier_uniform_(w)

        if self.args.fusion_type == 'static':
            nn.init.xavier_uniform_(self.weight_static)
            # parameters for SMN-dynamic (W1,W2,ts)
        if self.args.fusion_type == 'dynamic':
            linear1_weight = (param.data for name, param in self.linear1.named_parameters() if "weight" in name)
            for w in linear1_weight:
                nn.init.xavier_uniform_(w)

            linear2_weight = (param.data for name, param in self.linear2.named_parameters() if "weight" in name)
            for w in linear2_weight:
                nn.init.xavier_uniform_(w)

        final_linear_weight = (param.data for name, param in self.lin_output.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            nn.init.xavier_uniform_(w)
