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
user_dict_name = "/home/gong/zz/data/Match/dict_all.txt"
logger.info("loading {} user_dict".format(user_dict_name))
jieba.load_userdict(user_dict_name)

stopwords_file = "/home/gong/zz/data/Match/stop_words.txt"
logger.info("loading {} stop word".format(stopwords_file))
stopwords = {line.strip(): 0 for line in open(stopwords_file, 'r', encoding="utf-8").readlines()}


class Example(object):
    def __init__(self, text_a, text_b, char_a, char_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.char_a = char_a,
        self.char_b = char_b,
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 left_ids, right_ids, left_char_ids, right_char_ids, left_len, right_len, left_chars_len, right_chars_len, label):
        self.left_ids = left_ids
        self.right_ids = right_ids
        self.left_char_ids = left_char_ids
        self.right_char_ids = right_char_ids
        self.left_len = left_len
        self.right_len = right_len
        self.left_chars_len = left_chars_len
        self.right_chars_len = right_chars_len
        self.label = label


class MatchCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 max_char_size=50000,
                 min_freq=1,
                 specials=None):
        super(MatchCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data_aug.pt" if self.args.aug else "data.pt")
        self.field_text_file = os.path.join(args.data_dir, "field_text.pt")
        self.field_char_file = os.path.join(args.data_dir, "field_char.pt")

        self.max_vocab_size = max_vocab_size
        self.max_char_size = max_char_size
        self.min_freq = min_freq
        self.specials = specials
        # self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=None, special_tokens=self.specials),
                      "char": TextField(tokenize_fn=lambda x: [y for y in x], special_tokens=self.specials),
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

        logger.info("Build Vocab from {} ...".format(data_train_file))
        # 根据训练集来定制词表
        self.build_vocab(train_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        # test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     # "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_text_file))
        field_text = {"itos": self.field["text"].itos,
                         "stoi": self.field["text"].stoi,
                         "vocab_size": self.field["text"].vocab_size,
                         "specials": self.field["text"].specials
                         }
        logger.info("Saved text field to '{}'".format(self.field_char_file))
        field_char = {"itos": self.field["char"].itos,
                         "stoi": self.field["char"].stoi,
                         "vocab_size": self.field["char"].vocab_size,
                         "specials": self.field["char"].specials
                         }

        torch.save(field_text, self.field_text_file)
        torch.save(field_char, self.field_char_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    @timer
    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 text_a, text_b, label
        """
        if not os.path.isfile(data_file):
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
            line_items = [item for item in line.strip().split("\t") if item]
            text_a_tokens = self.tokenizer(line_items[1])
            text_b_tokens = self.tokenizer(line_items[2])
            lines.append({"text_a": " ".join(text_a_tokens), "text_b": " ".join(text_b_tokens), "label": line_items[3],
                          "char_a": "".join(text_a_tokens), "char_b": "".join(text_b_tokens)})

        logger.info("Read total {} examples from {}".format(len(lines), data_type.upper()))
        f.close()
        return lines

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"text_a":, "text_b":},...]
        """

        xs = [[x["text_a"], x["text_b"]] for x in data]
        self.field["text"].build_vocab(xs,
                                       min_freq=self.min_freq,
                                       max_size=self.max_vocab_size)

        cs = [[x["char_a"], x["char_b"]] for x in data]
        self.field["char"].build_vocab(cs,
                                       min_freq=self.min_freq,
                                       max_size=self.max_char_size)

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
                char_a=data["char_a"],
                char_b=data["char_b"],
                label=data["label"])
            )

        len_seq_left = np.array(len_seq_left)
        len_seq_right = np.array(len_seq_right)
        logger.info("left {} sequence length converge 95%".format(np.percentile(len_seq_left, 95)))
        logger.info("right {} sequence length converge 95%".format(np.percentile(len_seq_right, 95)))

        return examples

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
        all_left_id = torch.tensor([f.left_ids for f in features], dtype=torch.long)
        all_right_id = torch.tensor([f.right_ids for f in features], dtype=torch.long)
        all_left_char_id = torch.tensor([f.left_char_ids for f in features], dtype=torch.long)
        all_right_char_id = torch.tensor([f.right_char_ids for f in features], dtype=torch.long)
        all_left_len = torch.tensor([f.left_len for f in features], dtype=torch.long)
        all_right_len = torch.tensor([f.right_len for f in features], dtype=torch.long)
        all_left_char_len = torch.tensor([f.left_chars_len for f in features], dtype=torch.long)
        all_right_char_len = torch.tensor([f.right_chars_len for f in features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_left_id, all_right_id, all_left_len, all_right_len, all_left_char_id, all_right_char_id, all_left_char_len, all_right_char_len, all_label)

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
        words = jieba.cut(line_clean.strip())
        word_list = list(words)
        words_clean = []
        for word in word_list:
            if word not in stopwords:
                words_clean.append(word)

        # jieba.disable_parallel()
        return words_clean

    def convert_examples_to_features(self, examples, data_type="train"):
        features = []
        text_a_len, text_b_len = [], []
        chars_a_len, chars_b_len = [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for example in tqdm(examples, desc=desc_message):
            text_a_words = str.split(example.text_a)
            text_b_words = str.split(example.text_b)
            text_a_len.append(len(text_a_words))
            text_b_len.append(len(text_b_words))

            if len(text_a_words) > self.args.max_seq_length:
                text_a_words = text_a_words[:self.args.max_seq_length]
            left_len = len(text_a_words)

            if len(text_b_words) > self.args.max_seq_length:
                text_b_words = text_b_words[:self.args.max_seq_length]
            right_len = len(text_b_words)
            left_ids, right_ids, left_char_ids, right_char_ids = [], [], [], []
            char_a_len, char_b_len = [], []
            for i, word in enumerate(text_a_words):
                left_ids.append(
                    self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
                left_char_id = []
                for char in word:
                    left_char_id.append(
                        self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
                chars_a_len.append(len(left_char_id))
                if len(left_char_id) > self.args.max_char_seq_length:
                    left_char_id = left_char_id[:self.args.max_char_seq_length]
                left_char_ids.append(left_char_id)
                char_a_len.append(len(left_char_id))

            for i, word in enumerate(text_b_words):
                right_ids.append(
                    self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
                right_char_id = []
                for char in word:
                    right_char_id.append(
                        self.field["char"].stoi.get(char, self.field["char"].stoi.get(constants.UNK_WORD)))
                chars_b_len.append(len(right_char_id))
                if len(right_char_id) > self.args.max_char_seq_length:
                    right_char_id = right_char_id[:self.args.max_char_seq_length]
                right_char_ids.append(right_char_id)
                char_b_len.append(len(right_char_id))
            padding_id = self.field["text"].stoi[constants.PAD_WORD]
            left_ids = self.padding_seq(left_ids, self.args.max_seq_length, padding_id)
            right_ids = self.padding_seq(right_ids, self.args.max_seq_length, padding_id)

            char_padding_id = self.field["char"].stoi[constants.PAD_WORD]
            left_char_ids = self.padding_char_seq(left_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)
            right_char_ids = self.padding_char_seq(right_char_ids, self.args.max_seq_length, char_padding_id, self.args.max_char_seq_length)

            char_a_len = self.padding_seq(char_a_len, self.args.max_seq_length, padding_id)
            char_b_len = self.padding_seq(char_b_len, self.args.max_seq_length, padding_id)

            assert len(left_ids) == self.args.max_seq_length
            assert len(right_ids) == self.args.max_seq_length
            assert len(left_char_ids) == self.args.max_seq_length
            assert len(right_char_ids) == self.args.max_seq_length

            features.append(InputFeatures(
                left_ids=left_ids,
                right_ids=right_ids,
                left_char_ids=left_char_ids,
                right_char_ids=right_char_ids,
                left_len=left_len,
                right_len=right_len,
                left_chars_len=char_a_len,
                right_chars_len=char_b_len,
                label=int(example.label)
            ))

        text_a_len = np.array(text_a_len)
        text_b_len = np.array(text_b_len)
        chars_a_len = np.array(chars_a_len)
        chars_b_len = np.array(chars_b_len)
        print("{} sequence length converge 95%".format(np.percentile(text_a_len, 95)))
        print("{} sequence length converge 95%".format(np.percentile(text_b_len, 95)))
        print("{} char length converge 95%".format(np.percentile(chars_a_len, 95)))
        print("{} char length converge 95%".format(np.percentile(chars_b_len, 95)))
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