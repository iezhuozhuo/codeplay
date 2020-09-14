#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/metrics.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


def accuracy(logits, targets, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    _, preds = logits.max(dim=2)
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    acc = acc.mean()
    return acc


def attn_accuracy(logits, targets):
    """
    logits: (batch_size, vocab_size)
    targets: (batch_size)
    """
    _, preds = logits.squeeze(1).max(dim=-1)
    trues = (preds == targets).float()
    acc = trues.mean()
    return acc


def perplexity(logits, targets, weight=None, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1))
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def cosine(X, Y):
    """
    cosine
    """
    sim = np.sum(X * Y, axis=1) / \
        (np.sqrt((np.sum(X * X, axis=1) * np.sum(Y * Y, axis=1))) + 1e-10)
    return sim


class EmbeddingMetrics(object):
    """
    EmbeddingMetrics
    """
    def __init__(self, field):
        self.field = field
        assert field.embeddings is not None
        self.embeddings = np.array(field.embeddings)

    def texts2embeds(self, texts):
        """
        texts2embeds
        """
        texts = [self.field.numericalize(text)[1:-1] for text in texts]
        embeds = []
        for text in texts:
            vecs = self.embeddings[text]
            mask = vecs.any(axis=1)
            vecs = vecs[mask]
            if vecs.shape[0] == 0:
                vecs = np.zeros((1,) + vecs.shape[1:])
            embeds.append(vecs)
        return embeds

    def average(self, embeds):
        """
        average
        """
        avg_embeds = [embed.mean(axis=0) for embed in embeds]
        avg_embeds = np.array(avg_embeds)
        return avg_embeds

    def extrema(self, embeds):
        """
        extrema
        """
        ext_embeds = []
        for embed in embeds:
            s_max = np.max(embed, axis=0)
            s_min = np.min(embed, axis=0)
            s_plus = np.abs(s_min) <= s_max
            s = s_max * s_plus + s_min * np.logical_not(s_plus)
            ext_embeds.append(s)
        ext_embeds = np.array(ext_embeds)
        return ext_embeds

    def greedy(self, hyp_embeds, ref_embeds):
        """
        greedy
        """
        greedy_sim = []
        for hyp_embed, ref_embed in zip(hyp_embeds, ref_embeds):
            cos_sim = cosine_similarity(hyp_embed, ref_embed)
            g_sim = (cos_sim.max(axis=1).mean() +
                     cos_sim.max(axis=0).mean()) / 2
            greedy_sim.append(g_sim)
        greedy_sim = np.array(greedy_sim)
        return greedy_sim

    def embed_sim(self, hyp_texts, ref_texts):
        """
        embed_sim
        """
        assert len(hyp_texts) == len(ref_texts)
        hyp_embeds = self.texts2embeds(hyp_texts)
        ref_embeds = self.texts2embeds(ref_texts)

        ext_hyp_embeds = self.extrema(hyp_embeds)
        ext_ref_embeds = self.extrema(ref_embeds)
        ext_sim = cosine(ext_hyp_embeds, ext_ref_embeds)
        # print(len(ext_sim), (ext_sim > 0).sum())
        # print(ext_sim.sum() / (ext_sim > 0).sum())
        ext_sim_avg = ext_sim.mean()

        avg_hyp_embeds = self.average(hyp_embeds)
        avg_ref_embeds = self.average(ref_embeds)
        avg_sim = cosine(avg_hyp_embeds, avg_ref_embeds)
        # print(len(avg_sim), (avg_sim > 0).sum())
        # print(avg_sim.sum() / (avg_sim > 0).sum())
        avg_sim_avg = avg_sim.mean()

        greedy_sim = self.greedy(hyp_embeds, ref_embeds)
        # print(len(greedy_sim), (greedy_sim > 0).sum())
        # print(greedy_sim.sum() / (greedy_sim > 0).sum())
        greedy_sim_avg = greedy_sim.mean()

        return ext_sim_avg, avg_sim_avg, greedy_sim_avg


# 检索式对话常用的性能指标
def mean_average_precision(sort_data):
    # to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index + 1)
    return sum_precision / count_1


def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0


def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    return 1.0 * select_lable.count(1) / sort_lable.count(1)


def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5


def evaluate(file_path):
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    i = 0
    total_num = 0
    with open(file_path, 'r') as infile:
        for line in infile:
            if i % 10 == 0:
                data = []

            tokens = line.strip().split('\t')
            data.append((float(tokens[0]), int(tokens[1])))

            if i % 10 == 9:
                total_num += 1
                m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
                sum_m_a_p += m_a_p
                sum_m_r_r += m_r_r
                sum_p_1 += p_1
                sum_r_1 += r_1
                sum_r_2 += r_2
                sum_r_5 += r_5

            i += 1

    # print('total num: %s' %total_num)
    # print('MAP: %s' %(1.0*sum_m_a_p/total_num))
    # print('MRR: %s' %(1.0*sum_m_r_r/total_num))
    # print('P@1: %s' %(1.0*sum_p_1/total_num))
    return (1.0 * sum_m_a_p / total_num, 1.0 * sum_m_r_r / total_num, 1.0 * sum_p_1 / total_num,
            1.0 * sum_r_1 / total_num, 1.0 * sum_r_2 / total_num, 1.0 * sum_r_5 / total_num)


# MRC中的性能指标
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores