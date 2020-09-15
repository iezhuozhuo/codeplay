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


# 定义输入的Example类
class Example(object):
    def __init__(self, text_a, text_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# 定义输入feature类
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


# 定义任务的预料处理器 Processor
class ModelCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None):
        super(ModelCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data.pt")
        self.field_text_file = os.path.join(args.data_dir, "field_text.pt")
        self.max_vocab_size = max_vocab_size

        self.min_freq = min_freq
        self.specials = specials
        self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=self.tokenizer, special_tokens=self.specials),
                      "label": NumberField()}

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
            data: 字典列表，每个字典由 text_a, text_b, label
        """
        if not os.path.exists(data_file):
            logger.info("{} data can't find".format(data_type))
            return None

        f = open(data_file, 'r', encoding="utf-8")
        lines = []
        for i, line in enumerate(f):
            if i % 10000 == 0:
                logger.info("Read {} examples from {}".format(len(lines), data_type.upper()))
            # FIXME 全量数据
            # if len(lines) >= 20000:
            #     break
            # 读入数据可以自定义
            line_items = [item for item in line.strip().split("\t") if item]
            text_tokens = self.tokenizer(line_items[0])
            label = line_items[1]
            lines.append({"text": " ".join(text_tokens), "label": label})

        logger.info("Read total {} examples from {}".format(len(lines), data_type.upper()))
        f.close()
        return lines

    def build_examples(self, data_raw, data_type="train"):
        if data_raw == None:
            logger.info("{} data text and label can't find".format(data_type))

        examples, len_seq_left, len_seq_right = [], [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            len_seq_left.append(len(str.split(data["text_a"])))
            len_seq_right.append(len(str.split(data["text_b"])))
            examples.append(Example(
                text_a=data["text_a"],
                text_b=data["text_b"],
                label=data["label"])
            )

        len_seq_left = np.array(len_seq_left)
        len_seq_right = np.array(len_seq_right)
        logger.info("left {} sequence length converge 95%".format(np.percentile(len_seq_left, 95)))
        logger.info("right {} sequence length converge 95%".format(np.percentile(len_seq_right, 95)))

        return examples

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"text_a":, "text_b":},...]
        """

        xs = [[x["text_a"], x["text_b"]] for x in data]
        self.field["text"].build_vocab(xs,
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
        text_char_field = torch.load(self.field_char_file)
        self.field["text"].load(text_field)
        self.field["char"].load(text_char_field)

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
        dataset = None # 记得delete
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