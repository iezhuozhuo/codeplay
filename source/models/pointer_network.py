import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import LSTMEncoder
from source.modules.decoders.rnn_decoder import PointerDecoder
from source.modules.decoders.state import ReduceState


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

        self.reduce_state = ReduceState(self.hidden_size, self.bidirectional)
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
                                   rnn_hidden_size=self.hidden_size,
                                   dropout=self.dropout,
                                   output_type="seq2seq")

        self.decoder = PointerDecoder(input_size=self.input_size,
                                      hidden_size=self.hidden_size,
                                      vocab_size=self.n_vocab,
                                      embedder=self.embedder,
                                      num_layers=self.num_layers,
                                      dropout=self.dropout)

    # article_ids, article_len, article_mask,
    # summary_input_ids, summary_len, summary_taget_ids, summary_mask,
    # article_ids_extend_vocab = None, article_oovs = None, extra_zeros = None,
    def forward(self,
                article_ids, article_len, article_mask, article_ids_extend_vocab,
                summary_input_ids, summary_taget_ids, summary_mask, summary_len,
                h_context, extra_zeros, coverage,):

        encoder_outputs, encoder_feature, encoder_hidden = self.encoder((article_ids, article_len))

        s_t = self.reduce_state(encoder_hidden)
        step_losses = []
        for di in range(min(summary_len.cpu().numpy().max(), self.max_dec_steps)):
            y_t = summary_input_ids[:, di]  # 摘要的一个单词，batch里的每个句子的同一位置的单词编码
            # print("y_t_1:", y_t_1, y_t_1.size())
            final_dist, s_t, h_context, attn_weight, p_gen, coverage_next = self.decoder(y_t, s_t,
                                                                                         encoder_outputs,
                                                                                         encoder_feature, article_mask,
                                                                                         h_context,
                                                                                         extra_zeros,
                                                                                         article_ids_extend_vocab,
                                                                                         coverage, di)
            target = summary_taget_ids[:, di]  # 摘要的下一个单词的编码
            # print("target-iter:", target, target.size())
            # print("final_dist:", final_dist, final_dist.size())
            # input("go on>>")

            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表，也就是大于预设的50_000
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()  # 取出目标单词的概率gold_probs
            step_loss = -torch.log(gold_probs + self.eps)  # 最大化gold_probs，也就是最小化step_loss（添加负号）
            if self.is_coverage:
                # 当前attn_weight和先前coverage取最小
                step_coverage_loss = torch.sum(
                    torch.min(attn_weight, coverage.index_select(1, torch.arange(0, attn_weight.size(1), dtype=int).to(coverage.device))), 1)
                step_loss = step_loss + self.cov_loss_wt * step_coverage_loss
                coverage = coverage_next

            step_mask = summary_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / summary_len
        loss = torch.mean(batch_avg_loss)
        return loss
