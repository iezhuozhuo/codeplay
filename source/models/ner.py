# coding: utf-8

from itertools import zip_longest

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertModel


from source.models.crf import CRF, CRFOptimized
import source.utils.Constant as constants
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.losses.focal_loss import FocalLoss
from source.losses.label_smoothing import LabelSmoothingCrossEntropy


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
        outputs, last_hidden = self.rnn_encoder((inputs[0], inputs[1]), hidden)
        logits = self.hidden2tag(outputs)
        outputs = (logits,)

        if labels is not None:
            loss = self.crf.calculate_loss(features=logits, tag_list=labels, lengths=inputs[1])
            outputs = (loss,) + outputs

        return outputs


# another version birnn_crf
class BiRNN_CRFOptimized(nn.Module):
    def __init__(self,
                 args,
                 vocab_size,
                 tag_to_ix,
                 embedded_pretrain=None,
                 padding_idx=None,
                 dropout=0.5,
                 ):
        super(BiRNN_CRFOptimized, self).__init__()

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

        self.crf = CRFOptimized(
            num_tags=self.tagset_size,
            batch_first=True
            )

    def forward(self, inputs, labels=None, hidden=None):
        outputs, last_hidden = self.rnn_encoder(inputs[0], hidden)
        logits = self.hidden2tag(outputs)
        outputs = (logits,)

        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=None)
            outputs = (-1 * loss,) + outputs

        return outputs  # (loss), scores


# Bert 系列
class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRFOptimized(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores