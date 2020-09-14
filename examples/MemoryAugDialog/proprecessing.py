# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!
import os
import re
import json
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


class Example(object):
    def __init__(self, src=None, tgt=None, label=None,
                 post=None, resp=None, kg=None, kg_index=None):
        self.src = src
        self.tgt = tgt
        self.label = label

        self.post = post
        self.resp = resp
        self.kg = kg
        self.kg_index = kg_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_id,
                 input_len,
                 input_mask,
                 label):
        self.input_id = input_id
        self.input_len = input_len
        self.input_mask = input_mask
        self.label = label


class MemGMCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None):
        super(MemGMCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data.pt")
        self.field_src_file = os.path.join(args.data_dir, "field_src.pt")
        self.field_tgt_file = os.path.join(args.data_dir, "field_tgt.pt")
        self.max_vocab_size = max_vocab_size

        self.min_freq = min_freq
        self.specials = specials
        self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"src": TextField(tokenize_fn=None, special_tokens=self.specials),
                      "tgt": TextField(tokenize_fn=None, special_tokens=self.specials), }

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
        random.shuffle(train_raw)
        valid_raw = self.read_data(data_dev_file, data_type="valid")
        test_raw = self.read_data(data_dev_file, data_type="test")
        logger.info("Build Vocab from {} ...".format(data_train_file))
        # 根据训练集来定制词表
        self.build_vocab(train_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_src_file))
        field_text = {"itos": self.field["src"].itos,
                      "stoi": self.field["src"].stoi,
                      "vocab_size": self.field["src"].vocab_size,
                      "specials": self.field["src"].specials
                      }
        torch.save(field_text, self.field_src_file)

        logger.info("Saved text field to '{}'".format(self.field_tgt_file))
        field_text = {"itos": self.field["tgt"].itos,
                      "stoi": self.field["tgt"].stoi,
                      "vocab_size": self.field["tgt"].vocab_size,
                      "specials": self.field["tgt"].specials
                      }
        torch.save(field_text, self.field_tgt_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    @timer
    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 src, tgt
        """
        if not os.path.exists(data_file):
            logger.info("{} data can't find".format(data_type))
            return None

        f = open(data_file, 'r', encoding="utf-8", errors="ignore")
        lines = []
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                logger.info("Read {} examples from {}".format(len(lines), data_type.upper()))
            # FIXME 全量数据
            # if len(lines) >= 20000:
            #     break
            # 读入数据可以自定义
            line_items = [item for item in line.strip().split("\t") if item]
            if len(line_items) != 2:
                continue
            src, tgt = line_items
            if len(src) == 0 or len(tgt) == 0:
                print("miss: %s,%s" % (src, tgt))
                continue
            lines.append({"src": src, "tgt": tgt})

        logger.info("Read total {} examples from {}".format(len(lines), data_type.upper()))
        f.close()
        return lines

    def build_examples(self, data_raw, data_type="train"):
        if data_raw == None:
            logger.info("{} data text and label can't find".format(data_type))

        examples, src_len, tgt_len = [], [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            src_len.append(len(str.split(data["src"])))
            tgt_len.append(len(str.split(data["tgt"])))
            examples.append(Example(
                src=data["src"],
                tgt=data["tgt"]
            )
            )

        src_len = np.array(src_len)
        tgt_len = np.array(tgt_len)
        logger.info("left {} sequence length converge 95%".format(np.percentile(src_len, 95)))
        logger.info("right {} sequence length converge 95%".format(np.percentile(tgt_len, 95)))

        return examples

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"src":, "tgt":},...]
        """

        x_src = [x["src"] for x in data]
        x_tgt = [x["tgt"] for x in data]
        self.field["src"].build_vocab(x_src,
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)

        self.field["tgt"].build_vocab(x_tgt,
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        logger.info("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        # logger.info("Number of examples:",
        #       " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self):
        src_field = torch.load(self.field_src_file)
        tgt_field = torch.load(self.field_tgt_file)
        self.field["src"].load(src_field)
        self.field["tgt"].load(tgt_field)

    def create_batch(self, data_type="train"):
        examples = self.data[data_type]
        # FIXME Check example num
        # examples = examples[0:1024]
        features_cache_path = os.path.join(
            self.args.data_dir,
            "features-{}-{}-{}.pt".format(data_type, self.args.max_seq_length, "aug" if self.args.aug else "no-aug")
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading prepared features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert examples to features")
            features = self.convert_examples_to_features(examples)
            torch.save(features, features_cache_path)
        dataset = None  # 记得delete
        # 按需修改
        # all_left_id = torch.tensor([f.left_ids for f in features], dtype=torch.long)
        # all_right_id = torch.tensor([f.right_ids for f in features], dtype=torch.long)
        # all_left_char_id = torch.tensor([f.left_char_ids for f in features], dtype=torch.long)
        # all_right_char_id = torch.tensor([f.right_char_ids for f in features], dtype=torch.long)
        # all_left_len = torch.tensor([f.left_len for f in features], dtype=torch.long)
        # all_right_len = torch.tensor([f.right_len for f in features], dtype=torch.long)
        # all_left_char_len = torch.tensor([f.left_chars_len for f in features], dtype=torch.long)
        # all_right_char_len = torch.tensor([f.right_chars_len for f in features], dtype=torch.long)
        # all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_left_id, all_right_id, all_left_len, all_right_len, all_left_char_id, all_right_char_id, all_left_char_len, all_right_char_len, all_label)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def convert_examples_to_features(self, examples, data_type="train"):
        features = []
        text_a_len, text_b_len = [], []
        chars_a_len, chars_b_len = [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for example in tqdm(examples, desc=desc_message):
            pass
        #  数据读入按需修改
        #     text_a_words = str.split(example.text_a)
        #     text_b_words = str.split(example.text_b)
        #     text_a_len.append(len(text_a_words))
        #     text_b_len.append(len(text_b_words))
        #
        #     if len(text_a_words) > self.args.max_seq_length:
        #         text_a_words = text_a_words[:self.args.max_seq_length]
        #     left_len = len(text_a_words)
        #
        #     if len(text_b_words) > self.args.max_seq_length:
        #         text_b_words = text_b_words[:self.args.max_seq_length]
        #     right_len = len(text_b_words)
        #     inputs_ids = []
        #     char_a_len, char_b_len = [], []
        #     for i, word in enumerate(text_a_words):
        #         inputs_ids.append(
        #             self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
        #         left_char_id = []
        #         for char in word:
        #             left_char_id.append(
        #                 self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
        #         chars_a_len.append(len(left_char_id))
        #         if len(left_char_id) > self.args.max_char_seq_length:
        #             left_char_id = left_char_id[:self.args.max_char_seq_length]
        #         left_char_ids.append(left_char_id)
        #         char_a_len.append(len(left_char_id))
        #
        #     for i, word in enumerate(text_b_words):
        #         right_ids.append(
        #             self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
        #         right_char_id = []
        #         for char in word:
        #             right_char_id.append(
        #                 self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
        #         chars_b_len.append(len(right_char_id))
        #         if len(right_char_id) > self.args.max_char_seq_length:
        #             right_char_id = right_char_id[:self.args.max_char_seq_length]
        #         right_char_ids.append(right_char_id)
        #         char_b_len.append(len(right_char_id))
        #     padding_id = self.field["text"].stoi[constants.PAD_WORD]
        #     left_ids = self.padding_seq(left_ids, self.args.max_seq_length, padding_id)
        #     right_ids = self.padding_seq(right_ids, self.args.max_seq_length, padding_id)
        #
        #     char_padding_id = self.field["char"].stoi[constants.PAD_WORD]
        #     left_char_ids = self.padding_char_seq(left_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)
        #     right_char_ids = self.padding_char_seq(right_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)
        #
        #     assert len(left_ids) == self.args.max_seq_length
        #     assert len(right_ids) == self.args.max_seq_length
        #     assert len(left_char_ids) == self.args.max_seq_length
        #     assert len(right_char_ids) == self.args.max_seq_length
        #
        #     features.append(InputFeatures(
        #         input_id=None,
        #         input_len=None,
        #         input_mask=None,
        #         label=int(example.label)
        #     ))
        #
        # text_len = np.array(text_len)
        # print("{} sequence length converge 95%".format(np.percentile(text_len, 95)))

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

    def get_tokenizer(self):
        pass


class MemKdCorpus(object):
    def __init__(self,
                 args,
                 num_turns=8,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None):
        super(MemKdCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data.pt")
        self.field_text_file = os.path.join(args.data_dir, "field_text.pt")
        self.max_vocab_size = max_vocab_size

        self.min_freq = min_freq
        self.specials = specials
        self.tokenizer = self.get_tokenizer()
        self.num_turns = num_turns

        logger.info("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=None, special_tokens=self.specials, online=True)}

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
        train_example = self.read_data(data_train_file, data_type="train")
        random.shuffle(train_example)
        # valid_raw = self.read_data(data_dev_file, data_type="valid")
        # test_raw = self.read_data(data_dev_file, data_type="test")
        logger.info("Build Vocab from {} ...".format(data_train_file))
        # 根据训练集来定制词表
        self.build_vocab()

        # train_data = self.build_examples(train_raw, data_type="train")
        # valid_data = self.build_examples(valid_raw, data_type="valid")
        # test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_example,
                     # "valid": valid_data,
                     # "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_text_file))
        field_text = {"itos": self.field["text"].itos,
                      "stoi": self.field["text"].stoi,
                      "vocab_size": self.field["text"].vocab_size,
                      "specials": self.field["text"].specials
                      }
        torch.save(field_text, self.field_text_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    @timer
    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            example: post, resp, kg, kg_index
        """
        if not os.path.exists(data_file):
            logger.info("{} data can't find".format(data_type))
            return None

        examples = []
        # origin_data[data_type] = {'post': [], 'resp': [], 'kg': [], 'kg_index': []}
        datas = json.load(open(data_file, 'r', encoding="utf-8", errors="ignore"))
        for i, data in enumerate(datas):
            if i % 100 == 0:
                logger.info("Read {} examples from {}".format(len(examples), data_type.upper()))

            messages = data['messages']
            kg, kg_index = [], []
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
                        # count_token(h + r + t)
                        self.field["text"].online_update_vocab(h + r + t)
            history = []
            i = 0
            nxt_sent = jieba.lcut(messages[0]['message'])
            while i + 1 < len(messages):
                tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                history.append(tmp_sent)
                post = []
                for jj in range(max(-self.num_turns + 1, -i - 1), 0):
                    post = post + history[jj] + [constants.EOS_WORD, constants.GO]
                post = post[:-2]

                # count_token(nxt_sent)
                self.field["text"].online_update_vocab(nxt_sent)
                if i == 0:
                    # count_token(tmp_sent)
                    self.field["text"].online_update_vocab(tmp_sent)
                examples.append(Example(
                    post=post,
                    resp=nxt_sent,
                    kg=kg,
                    kg_index=kg_index[i + 1]
                ))
                # origin_data[data_type]['post'].append(post)
                # origin_data[data_type]['resp'].append(nxt_sent)
                # origin_data[data_type]['kg'].append(kg)
                # origin_data[data_type]['kg_index'].append(kg_index[i + 1])
                i += 1
        return examples

    def build_vocab(self):
        """
        从train的text分别生成字典
        data format [{"src":, "tgt":},...]
        """
        self.field["text"].build_vocab(None,
                                       min_freq=self.min_freq,
                                       max_size=self.max_vocab_size)

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
            "features-{}-{}-{}.pt".format(data_type, self.args.max_seq_length, "aug" if self.args.aug else "no-aug")
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading prepared features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert examples to features")
            features = self.convert_examples_to_features(examples)
            torch.save(features, features_cache_path)
        dataset = None  # 记得delete
        # 按需修改
        # all_left_id = torch.tensor([f.left_ids for f in features], dtype=torch.long)
        # all_right_id = torch.tensor([f.right_ids for f in features], dtype=torch.long)
        # all_left_char_id = torch.tensor([f.left_char_ids for f in features], dtype=torch.long)
        # all_right_char_id = torch.tensor([f.right_char_ids for f in features], dtype=torch.long)
        # all_left_len = torch.tensor([f.left_len for f in features], dtype=torch.long)
        # all_right_len = torch.tensor([f.right_len for f in features], dtype=torch.long)
        # all_left_char_len = torch.tensor([f.left_chars_len for f in features], dtype=torch.long)
        # all_right_char_len = torch.tensor([f.right_chars_len for f in features], dtype=torch.long)
        # all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_left_id, all_right_id, all_left_len, all_right_len, all_left_char_id, all_right_char_id, all_left_char_len, all_right_char_len, all_label)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def convert_examples_to_features(self, examples, data_type="train"):
        """
        :param examples:
        post:对话历史  中间有<\s> <go>连接区分每句话
        resp：当前post的回复
        kg 对话历史以及post用到的所有知识
        kg_index 该对话用到的知识index
        """
        features = []
        text_a_len, text_b_len = [], []
        chars_a_len, chars_b_len = [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for example in tqdm(examples, desc=desc_message):
            pass
        #  数据读入按需修改
        #     text_a_words = str.split(example.text_a)
        #     text_b_words = str.split(example.text_b)
        #     text_a_len.append(len(text_a_words))
        #     text_b_len.append(len(text_b_words))
        #
        #     if len(text_a_words) > self.args.max_seq_length:
        #         text_a_words = text_a_words[:self.args.max_seq_length]
        #     left_len = len(text_a_words)
        #
        #     if len(text_b_words) > self.args.max_seq_length:
        #         text_b_words = text_b_words[:self.args.max_seq_length]
        #     right_len = len(text_b_words)
        #     inputs_ids = []
        #     char_a_len, char_b_len = [], []
        #     for i, word in enumerate(text_a_words):
        #         inputs_ids.append(
        #             self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
        #         left_char_id = []
        #         for char in word:
        #             left_char_id.append(
        #                 self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
        #         chars_a_len.append(len(left_char_id))
        #         if len(left_char_id) > self.args.max_char_seq_length:
        #             left_char_id = left_char_id[:self.args.max_char_seq_length]
        #         left_char_ids.append(left_char_id)
        #         char_a_len.append(len(left_char_id))
        #
        #     for i, word in enumerate(text_b_words):
        #         right_ids.append(
        #             self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
        #         right_char_id = []
        #         for char in word:
        #             right_char_id.append(
        #                 self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
        #         chars_b_len.append(len(right_char_id))
        #         if len(right_char_id) > self.args.max_char_seq_length:
        #             right_char_id = right_char_id[:self.args.max_char_seq_length]
        #         right_char_ids.append(right_char_id)
        #         char_b_len.append(len(right_char_id))
        #     padding_id = self.field["text"].stoi[constants.PAD_WORD]
        #     left_ids = self.padding_seq(left_ids, self.args.max_seq_length, padding_id)
        #     right_ids = self.padding_seq(right_ids, self.args.max_seq_length, padding_id)
        #
        #     char_padding_id = self.field["char"].stoi[constants.PAD_WORD]
        #     left_char_ids = self.padding_char_seq(left_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)
        #     right_char_ids = self.padding_char_seq(right_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)
        #
        #     assert len(left_ids) == self.args.max_seq_length
        #     assert len(right_ids) == self.args.max_seq_length
        #     assert len(left_char_ids) == self.args.max_seq_length
        #     assert len(right_char_ids) == self.args.max_seq_length
        #
        #     features.append(InputFeatures(
        #         input_id=None,
        #         input_len=None,
        #         input_mask=None,
        #         label=int(example.label)
        #     ))
        #
        # text_len = np.array(text_len)
        # print("{} sequence length converge 95%".format(np.percentile(text_len, 95)))

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

    def get_tokenizer(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default="/home/gong/zz/data/weibo",
        type=str,
    )
    parser.add_argument(
        "--train_file",
        default="weibo_src_tgt_utf8.train",
        type=str,
    )
    parser.add_argument(
        "--dev_file",
        default="weibo_src_tgt_utf8.dev",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/home/gong/zz/data/weibo",
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
    specials = [constants.PAD_WORD, constants.UNK_WORD, constants.EOS_WORD, constants.GO]
    # processor = MemGMCorpus(args=args, specials=specials)
    processor = MemKdCorpus(args=args, specials=specials)
    # print(processor.field["src"].vocab_size)
