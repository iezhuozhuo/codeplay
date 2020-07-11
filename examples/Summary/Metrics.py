# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 21:01
# @Author  : zhuo & zdy
# @github   : iezhuozhuo

import os
import time
from rouge import Rouge

import torch
from torch.autograd import Variable

import source.utils.Constant as constant
from source.utils.misc import timer


# Beam Search
# context, coverage 用于pointerNet上
class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 state,
                 context=None,
                 coverage=None):
        super(Beam, self).__init__()

        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            context=context,
            coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, args, model, test_dataset, origin_dataset, vocab, logger):
        self.args = args
        self.model = model
        self.test_dataset = test_dataset
        self.origin_dataset = origin_dataset
        self.vocab = vocab
        self.logger = logger

        # self._decode_dir = os.path.join(args.output_dir, 'decode_file')
        # self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        # self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        # for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
        #     if not os.path.exists(p):
        #         os.mkdir(p)

        self.rouge = Rouge()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    @timer
    def decode(self):
        start = time.time()
        counter = 0
        self.model.eval()
        pred_summary, glod_summary = [], []
        for batch_id, (batch, origin_dataset) in enumerate(zip(self.test_dataset, self.origin_dataset), 1):
            # Run beam search to get best Hypothesis
            batch = tuple(t.to(self.args.device) for t in batch)
            article_ids, article_len, article_mask, article_ids_extend_vocab, extra_zeros = batch
            h_context = Variable(torch.zeros((article_ids.size(0), 2 * self.args.hidden_size))).to(self.args.device)
            coverage = Variable(torch.zeros(article_ids.size())).to(self.args.device)

            extra_zeros = extra_zeros.squeeze().expand(self.args.beam_size, self.args.max_oov_len).contiguous()
            best_summary = self.beam_search(article_ids, article_len, article_mask, article_ids_extend_vocab,
                                            h_context, extra_zeros, coverage)
            art_oovs, original_summarys = origin_dataset
            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = self.outputids2words(output_ids,
                                                 (art_oovs if self.args.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(constant.EOS_WORD)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            pred_summary.append(" ".join(decoded_words))
            glod_summary.append(original_summarys)

            counter += 1
            if counter % 100 == 0:
                self.logger.info('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()

        self.logger.info("Beam search has finished.")
        self.logger.info("Now starting ROUGE eval...")
        metrics = self.rouge.get_scores(pred_summary, glod_summary, avg=True)
        for key in sorted(metrics.keys()):
            self.logger.info("{}: {}".format(key.upper(), metrics[key]))

    def beam_search(self,
                    article_ids, article_len, article_mask, article_ids_extend_vocab,
                    h_context, extra_zeros, coverage):
        # batch should have only one example

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder((article_ids, article_len))
        encoder_outputs = encoder_outputs.squeeze().expand(
            self.args.beam_size, encoder_outputs.size(1), encoder_outputs.size(2)).contiguous()
        # encoder_outputs = torch.stack([encoder_outputs.squeeze() for _ in range(self.args.beam_size)], 0)
        encoder_feature = torch.cat([encoder_feature for _ in range(self.args.beam_size)], 0)

        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.stoi[constant.BOS_WORD]],
                      log_probs=[0.0],
                      state=(dec_h, dec_c),
                      context=h_context[0],
                      coverage=coverage[0])
                 for _ in range(self.args.beam_size)]
        results = []
        steps = 0
        while steps < self.args.max_dec_seq_length and len(results) < self.args.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.vocab_size else self.vocab.stoi[constant.UNK_WORD] \
                             for t in latest_tokens]
            y_t = Variable(torch.LongTensor(latest_tokens)).to(encoder_outputs.device)
            all_state_h, all_state_c, all_context = [], [], []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            h_context = torch.stack(all_context, 0)

            coverage_t = None
            if self.args.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t = torch.stack(all_coverage, 0)

            final_dist, s_t, h_context, attn_weight, p_gen, coverage_next = self.model.decoder(y_t, s_t,
                                                                                    encoder_outputs, encoder_feature,
                                                                                    article_mask, h_context,
                                                                                    extra_zeros, article_ids_extend_vocab,
                                                                                    coverage_t, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.args.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = h_context[i]
                coverage_i = (coverage_next[i] if self.args.is_coverage else None)

                for j in range(self.args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.stoi[constant.EOS_WORD]:
                    if steps >= self.args.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.args.beam_size or len(results) == self.args.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def outputids2words(self, id_list, article_oovs):
        words = []
        for i in id_list:
            try:
                w = self.vocab.itos[i]  # might be [UNK]
            except IndexError as e:  # w is OOV
                assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
                article_oov_idx = i - self.vocab.vocab_size
                try:
                    w = article_oovs[article_oov_idx]
                except ValueError as e:  # i doesn't correspond to an article oov
                    raise ValueError(
                        'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                            i, article_oov_idx, len(article_oovs)))
            words.append(w)
        return words