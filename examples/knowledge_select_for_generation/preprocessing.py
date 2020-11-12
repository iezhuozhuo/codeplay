# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 14:22
# @Author  : zhuo & zdy
# @github   : iezhuozhuo

import os
import json
import re
import jieba
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from source.utils.misc import init_logger, timer
import source.utils.Constant as constants
from source.inputters.field import TextField, NumberField

logger = init_logger()
# user_dict_name = "/home/gong/zz/data/Match/dict_all.txt"
# logger.info("loading {} user_dict".format(user_dict_name))
# jieba.load_userdict(user_dict_name)
#
# stopwords_file = "/home/gong/zz/data/Match/stop_words.txt"
# logger.info("loading {} stop word".format(stopwords_file))
# stopwords = {line.strip(): 0 for line in open(stopwords_file, 'r', encoding="utf-8").readlines()}

dic = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}

class Example(object):
    def __init__(self, post, resp, kg, kg_index):
        self.post = post
        self.resp = resp
        self.kg = kg
        self.kg_index = kg_index

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 post_length, prev_length, resp_length, post, resp, post_allvocabs, resp_allvocabs, kg_h_length, kg_hr_length, kg_hrt_length, kg, kg_index):
        self.post_length = post_length
        self.prev_length = prev_length
        self.resp_length = resp_length
        self.post = post
        self.resp = resp
        self.post_allvocabs = post_allvocabs
        self.resp_allvocabs = resp_allvocabs
        self.kg_h_length = kg_h_length
        self.kg_hr_length = kg_hr_length
        self.kg_hrt_length = kg_hrt_length
        self.kg = kg
        self.kg_index = kg_index

class MatchCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 max_char_size=50000,
                 min_freq=1,
                 specials=None,
                 num_turns=8,
                 max_sent_length=10086,
                 max_know_length=100):
        super(MatchCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data.pt")
        self.field_text_file = os.path.join(args.data_dir, "field_text.pt")
        self.field_char_file = os.path.join(args.data_dir, "field_char.pt")

        self.max_vocab_size = max_vocab_size
        self.max_char_size = max_char_size
        self.min_freq = min_freq
        self.specials = specials
        self.vocab = {}
        self.num_turns = num_turns
        # self.tokenizer = self.get_tokenizer()
        self.pad_id = 0
        self.unk_id = 1
        self.go_id = 2
        self.eos_id = 3
        self.max_sent_length = max_sent_length
        self.max_know_length = max_know_length


        logger.info("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=None, special_tokens=self.specials)}

        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            logger.info("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field()

    def build(self):
        data_train_file = os.path.join(self.args.data_dir, self.args.train_file)
        data_dev_file = os.path.join(self.args.data_dir, self.args.dev_file)

        logger.info("Reading Data ...")
        train_raw = self.read_data(data_train_file, data_type="train")
        valid_raw = self.read_data(data_dev_file, data_type="valid")

        logger.info("Build Vocab from {} ...".format(data_train_file))
        # 根据训练集来定制词表
        self.build_vocab(train_raw)

        # train_data = self.build_examples(train_raw, data_type="train")
        # valid_data = self.build_examples(valid_raw, data_type="valid")
        # test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_raw,
                     "valid": valid_raw,
                     # "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_text_file))
        field_article = {"itos": self.field["text"].itos,
                         "stoi": self.field["text"].stoi,
                         "vocab_size": self.field["text"].vocab_size,
                         "specials": self.field["text"].specials
                         }



        torch.save(field_article, self.field_text_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    @timer
    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 text_a, text_b, label
        """
        def count_token(tokens):
            for token in tokens:
                self.vocab[token] = self.vocab[token] + 1 if token in self.vocab else 1

        if not os.path.isfile(data_file):
            logger.info("{} data can't find".format(data_type))
            return None

        origin_data = {'post': [], 'resp': [], 'kg': [], 'kg_index': []}
        datas = json.load(open(data_file, encoding='utf8'))

        for data in datas:
            messages = data['messages']
            kg = []
            kg_index = []
            kg_dict = {}
            for message in messages:
                kg_index.append([])
                if 'attrs' in message:
                    for attr in message['attrs']:
                        h = jieba.lcut(attr['name'])
                        r = jieba.lcut(attr['attrname'])
                        t = jieba.lcut(attr['attrvalue'])
                        k = tuple((tuple(h), tuple(r), tuple(t)))
                        if k not in kg_dict:
                            kg_dict[k] = len(kg)
                            kg.append(k)
                        kg_index[-1].append(kg_dict[k])
                        count_token(h + r + t)

            history = []
            i = 0
            nxt_sent = jieba.lcut(messages[0]['message'])
            while i + 1 < len(messages):
                tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                history.append(tmp_sent)
                post = []
                for jj in range(max(-self.num_turns + 1, -i - 1), 0):
                    post = post + history[jj] + ['<eos>', '<go>']
                post = post[:-2]

                count_token(nxt_sent)
                if i == 0:
                    count_token(tmp_sent)
                # origin_data.append(
                #     {"post": post, "resp": nxt_sent, "kg": kg, "kg_index": kg_index[i + 1]}
                # )
                origin_data['post'].append(post)
                origin_data['resp'].append(nxt_sent)
                origin_data['kg'].append(kg)
                origin_data['kg_index'].append(kg_index[i + 1])

                i += 1



        # lines = []
        #
        # with open(data_file) as f:
        #     next(f)  # skip the header row
        #     for line in f:
        #         sents = line.strip().split('\t')
        #         if sents[0] is '-':
        #             continue
        #
        #         words_in_left = sents[1].strip().split(' ')
        #         words_in_left = [x for x in words_in_left if x not in ('(', ')')]
        #
        #
        #         words_in_right = sents[2].strip().split(' ')
        #         words_in_right = [x for x in words_in_right if x not in ('(', ')')]
        #
        #         label = dic[sents[0]]
        #         lines.append(
        #             {"text_a": " ".join(words_in_left), "text_b": " ".join(words_in_right), "label": label,
        #              "char_a": "".join(words_in_left), "char_b": "".join(words_in_right)})




        logger.info("Read total {} examples from {}".format(len(origin_data['post']), data_type.upper()))
        return origin_data

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"text_a":, "text_b":},...]
        """
        # Important: Sort the words preventing the index changes between different runs
        vocab = sorted(list(self.vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self.min_freq, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.specials + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(lambda x: x[1] >= 0 and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        word2id = {w: i for i, w in enumerate(vocab_list)}

        self.field["text"].itos = vocab_list
        self.field["text"].stoi = word2id
        self.field["text"].vocab_size = len(word2id)
        self.field["text"].specials = self.specials

        # xs = [[x["text_a"], x["text_b"]] for x in data]
        # self.field["text"].build_vocab(xs,
        #                                min_freq=self.min_freq,
        #                                max_size=self.max_vocab_size)
        #
        # cs = [[x["char_a"], x["char_b"]] for x in data]
        # self.field["char"].build_vocab(cs,
        #                                min_freq=self.min_freq,
        #                                max_size=self.max_char_size)

    def build_examples(self, data_raw, data_type="train"):
        # if data_raw == None:
        #     logger.info("{} data text and label can't find".format(data_type))
        # examples, len_seq_left, len_seq_right = [], [], []
        # desc_message = "GETDATA FROM " + data_type.upper()
        # for data in tqdm(data_raw, desc=desc_message):
        #     len_seq_left.append(len(data["post"]))
        #     len_seq_right.append(len(str.split(data["resp"])))
        #     examples.append(Example(
        #         post=data["post"],
        #         resp=data["resp"],
        #         kg=data["kg"],
        #         char_b=data["char_b"]
        #         )
        #     )
        #
        # len_seq_left = np.array(len_seq_left)
        # len_seq_right = np.array(len_seq_right)
        # logger.info("left {} sequence length converge 95%".format(np.percentile(len_seq_left, 95)))
        # logger.info("right {} sequence length converge 95%".format(np.percentile(len_seq_right, 95)))

        return #examples

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        logger.info("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        # logger.info("Number of examples:",
        #       " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self):
        text_field = torch.load(self.field_text_file)
        self.field["text"].load(text_field)

    def create_batch(self, data_type="train"):
        examples = self.data[data_type]
        # FIXME Check example num
        # examples = examples[0:1024]
        features_cache_path = os.path.join(
            self.args.data_dir,
            "features-{}-{}-{}".format(data_type, self.args.max_seq_length, "no-aug")
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading prepared features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert examples to features")
            features = self.convert_examples_to_features(examples)
            torch.save(features, features_cache_path)

        post_length = torch.tensor([f.post_length for f in features], dtype=torch.long)
        prev_length = torch.tensor([f.prev_length for f in features], dtype=torch.long)
        resp_length = torch.tensor([f.resp_length for f in features], dtype=torch.long)
        post = torch.tensor([f.post for f in features], dtype=torch.long)
        resp = torch.tensor([f.resp for f in features], dtype=torch.long)
        post_allvocabs = torch.tensor([f.post_allvocabs for f in features], dtype=torch.long)
        resp_allvocabs = torch.tensor([f.resp_allvocabs for f in features], dtype=torch.long)
        kg_h_length = torch.tensor([f.kg_h_length for f in features], dtype=torch.long)
        kg_hr_length = torch.tensor([f.kg_hr_length for f in features], dtype=torch.long)
        kg_hrt_length = torch.tensor([f.kg_hrt_length for f in features], dtype=torch.long)
        kg = torch.tensor([f.kg for f in features], dtype=torch.long)
        kg_index = torch.tensor([f.kg_index for f in features], dtype=torch.long)

        dataset = TensorDataset(post_length, prev_length, resp_length, post, resp, post_allvocabs, resp_allvocabs, kg_h_length, kg_hr_length, kg_hrt_length, kg, kg_index)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def tokenizer(self, line):
        # jieba.enable_parallel()
        line_clean = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", line)
        # words = jieba.cut(line_clean.strip())
        # word_list = list(words)
        # words_clean = []
        # for word in word_list:
        #     if word not in stopwords:
        #         words_clean.append(word)
        #
        # # jieba.disable_parallel()
        return #words_clean

    def convert_examples_to_features(self, examples, data_type="train"):


        word2id = self.field['text'].stoi
        know2id = lambda line: list(map(lambda word: word2id.get(word, self.unk_id), line))
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + [
            self.eos_id])[:self.max_sent_length]
        knows2id = lambda lines: list(map(know2id, lines))

        data = {}
        data['post'] = list(map(line2id, examples['post']))
        data['resp'] = list(map(line2id, examples['resp']))
        data['kg'] = [list(map(knows2id, kg)) for kg in examples['kg']]
        data['kg_index'] = examples['kg_index']
        data_size = len(data['post'])
        self.valid_vocab_len = len(self.field["text"].itos)

        res = {}
        batch_size = data_size
        indexes = list(range(data_size))
        res["post_length"] = list(map(lambda i: len(data['post'][i]), indexes))
        res["prev_length"] = list(map(lambda i: (
            len(data['post'][i]) - data['post'][i][::-1].index(self.eos_id, 1)
            if self.eos_id in data['post'][i][:-1] else 0), indexes))


        res["resp_length"] = list(map(lambda i: len(data['resp'][i]), indexes))

        #padding
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            post = data['post'][idx]
            resp = data['resp'][idx]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post"] = res["post"].tolist()
        res["resp"] = res["resp"].tolist()
        res_post_list = res_post.tolist()
        res_resp_list = res_resp.tolist()
        res["post_allvocabs"] = res_post_list
        res["resp_allvocabs"] = res_resp_list


        # res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        # res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        # for i, idx in enumerate(indexes):
        #     post = data['post'][idx]
        #     resp = data['resp'][idx]
        #     res_post[i, :len(post)] = post
        #     res_resp[i, :len(resp)] = resp
        #
        # res["post_allvocabs"] = res_post.copy()
        # res["resp_allvocabs"] = res_resp.copy()


        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id




        max_kg_num = max([len(data['kg'][idx]) for idx in indexes])

        kg_h_length_all = np.zeros((batch_size, max_kg_num), dtype=int)
        kg_hr_length_all = np.zeros((batch_size, max_kg_num), dtype=int)
        kg_hrt_length_all = np.zeros((batch_size, max_kg_num), dtype=int)


        for i, idx in enumerate(indexes):
            kg_h_length = [min(self.max_know_length, len(sent[0])) for sent in data['kg'][idx]]
            kg_h_length_all[i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self.max_know_length, len(sent[0]) + len(sent[1])) for sent in data['kg'][idx]]
            kg_hr_length_all[i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self.max_know_length, len(sent[0]) + len(sent[1]) + len(sent[2])) for sent in
                             data['kg'][idx]]
            kg_hrt_length_all[i, :len(kg_hrt_length)] = kg_hrt_length

        kg_h_length_all = kg_h_length_all.tolist()
        kg_hr_length_all = kg_hr_length_all.tolist()
        kg_hrt_length_all = kg_hrt_length_all.tolist()

        res["kg_h_length"] = kg_h_length_all
        res["kg_hr_length"] = kg_hr_length_all
        res['kg_hrt_length'] = kg_hrt_length_all

        kg_all = np.zeros((batch_size, max_kg_num, np.max(res["kg_hrt_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            for j, tri in enumerate(data['kg'][idx]):
                sent = (tri[0] + tri[1] + tri[2])[:self.max_know_length]
                kg_all[i, j, :len(sent)] = sent
        res['kg'] = kg_all.tolist()

        kg_index_all = np.zeros((batch_size, max_kg_num), dtype=float)
        for i, idx in enumerate(indexes):
            for kgid in data['kg_index'][idx]:
                kg_index_all[i, kgid] = 1
        res['kg_index'] = kg_index_all.tolist()
        print('1')

        features = []
        for i in range(data_size):
            features.append(InputFeatures(
                post_length = res['post_length'][i],
                prev_length = res['prev_length'][i],
                resp_length = res['resp_length'][i],
                post = res['post'][i],
                resp = res['resp'][i],
                post_allvocabs = res['post_allvocabs'][i],
                resp_allvocabs = res['resp_allvocabs'][i],
                kg_h_length = res['kg_h_length'][i],
                kg_hr_length = res['kg_hr_length'][i],
                kg_hrt_length = res['kg_hrt_length'][i],
                kg = res['kg'][i],
                kg_index = res['kg_index'][i]
                ))
        print("{} sequence post_length length converge 95%".format(np.percentile(res['post_length'], 95)))
        print("{} sequence prev_length length converge 95%".format(np.percentile(res['prev_length'], 95)))
        print("{} sequence prev_length length converge 95%".format(np.percentile(res['prev_length'], 95)))

        return features

    def padding_seq(self, seq, max_len, pad_id):
        padding_length = max_len - len(seq)
        seq += [pad_id] * padding_length
        return seq

    def padding_char_seq(self, seq, max_len, pad_id, max_char_len):
        for i in range(len(seq)):
            padding_char_length = max_char_len - len(seq[i])
            seq[i] += [pad_id] * padding_char_length
        padding_length = max_len - len(seq)
        seq += [[pad_id for i in range(max_char_len)]] * padding_length
        return seq

    def padding_char_len(self, seq, max_len, pad_id):
        padding_length = max_len - len(seq)
        seq += [pad_id] * padding_length
        return seq


if __name__ == "__main__":
    import argparse
    # from source.utils.misc import checkoutput_and_setcuda

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--train_file",
        default="train.csv",
        type=str,
    )
    parser.add_argument(
        "--dev_file",
        default="dev.csv",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--max_char_seq_length",
        default=5,
        type=int,
    )
    parser.add_argument("--aug", action="store_true")

    args, _ = parser.parse_known_args()
    args.local_rank = -1
    # args = checkoutput_and_setcuda(args)
    specials = [constants.UNK_WORD, constants.PAD_WORD]
    processor = MatchCorpus(args=args, specials=specials)
    print(processor.field["text"].vocab_size)
    # s = "蚂蚁花呗 今天 要还  好 凄惨 啊"
    # print(processor.tokenizer(s))
    from source.modules.embedder import Embedder

    padding_idx = processor.field["text"].stoi[constants.PAD_WORD]
    embedding = Embedder(num_embeddings=processor.field["text"].vocab_size,
                         embedding_dim=128, padding_idx=padding_idx)
    embedding.load_embeddingsfor_gensim_vec("/home/gong/zz/data/Match/word2vec.model", processor.field["text"].stoi)
    args.train_batch_size = 64
    dataloader = processor.create_batch("train")