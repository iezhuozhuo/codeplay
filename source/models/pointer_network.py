import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import LSTMEncoder, RNNEncoder
from source.modules.attention import PointerAttention
trunc_norm_init_std = 1e-4
rand_unif_init_mag = 0.02


def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)


class ReduceState(nn.Module):
    def __init__(self, hidden_size):
        super(ReduceState, self).__init__()
        self.hidden_size = hidden_size
        self.reduce_h = nn.Linear(self.hidden_size * 2, self.hidden_size)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(self.hidden_size * 2, self.hidden_size)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden    # h, c dim = [2, batch, hidden_dim]
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)  # [batch, hidden_dim*2]
        hidden_reduced_h = F.relu(self.reduce_h(h_in))                         # [batch, hidden_dim]
        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = [1, batch, hidden_dim]


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 vocab_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 pointer_gen=True):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.pointer_gen = pointer_gen
        self.vocab_size = vocab_size

        self.attention_network = PointerAttention(self.hidden_size)
        # decoder
        self.x_context = nn.Linear(self.hidden_size * 2 + self.input_size, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if self.pointer_gen:
            self.p_gen_linear = nn.Linear(self.hidden_size * 4 + self.hidden_size, 1)

        #p_vocab
        self.out1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.out2 = nn.Linear(self.hidden_size, self.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_size),
                                 c_decoder.view(-1, self.hidden_size)), 1)  # B x 2*hidden_dim
            # attention
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedder(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_size),
                             c_decoder.view(-1, self.hidden_size)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hidden_size), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class PointerNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_vocab,
                 embedded_pretrain=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 padding_idx=None,
                 max_dec_steps=40
                 ):
        super(PointerNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab


        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.max_dec_steps = max_dec_steps
        self.eps = 1e-12
        self.cov_loss_wt = 1.0
        self.is_coverage = True

        self.reduce_state = ReduceState(self.hidden_size)
        self.embedder = Embedder(num_embeddings=self.n_vocab,
                                 embedding_dim=self.input_size, padding_idx=self.padding_idx)
        # init_wt_normal(self.embedder.weight)

        if embedded_pretrain is not None:
            self.embedder.load_embeddings(embedded_pretrain)

        self.encoder = LSTMEncoder(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              embedder=self.embedder,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout)

        # self.encoder = RNNEncoder(input_size=self.input_size,
        #                       hidden_size=self.hidden_size,
        #                       embedder=self.embedder,
        #                       num_layers=self.num_layers,
        #                       bidirectional=self.bidirectional,
        #                       dropout=self.dropout)

        self.decoder = Decoder(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              vocab_size=self.n_vocab,
                              embedder=self.embedder,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout)

    def forward(self, enc_batch, enc_lens, dec_batch, enc_padding_mask, c_t_1,
                                 extra_zeros, enc_batch_extend_vocab, coverage, target_batch, max_dec_len, dec_padding_mask):

        encoder_outputs, encoder_feature, encoder_hidden = self.encoder((enc_batch, enc_lens))

        # encoder_outputs, encoder_hidden = self.encoder((enc_batch, enc_lens))

        s_t_1 = self.reduce_state(encoder_hidden)
        step_losses = []
        for di in range(min(max(max_dec_len.cpu().numpy()), self.max_dec_steps)):
            y_t_1 = dec_batch[:, di]      # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]  # 摘要的下一个单词的编码
            # print("target-iter:", target, target.size())
            # print("final_dist:", final_dist, final_dist.size())
            # input("go on>>")
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的50_000
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()   # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + self.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if self.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + self.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        return step_losses
