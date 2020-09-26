# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!
import os
import re
import json
import random
import codecs
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from source.utils.misc import init_logger, timer
import source.utils.Constant as constants
from source.inputters.field import TextField, NumberField

logger = init_logger()

# 定义输入的Example类
class MTRSExample(object):
    def __init__(self, guid, utterences, response, label):
        self.guid = guid
        self.utterences = utterences
        self.response = response
        self.label = label


# 定义输入feature类
class MTRSFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 utters_id,
                 utters_len,
                 utters_num,
                 response_id,
                 response_len,
                 label=None):
        self.utters_id = utters_id
        self.utters_len = utters_len
        self.utters_num = utters_num
        self.response_id = response_id
        self.response_len = response_len
        self.label = label


# 定义任务的预料处理器 Processor
class MTRSCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None,
                 share_vocab=True):
        super(MTRSCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.output_dir, "data.pt")
        self.field_context_file = os.path.join(args.output_dir, "field_context.pt")
        self.field_response_file = os.path.join(args.output_dir, "field_response.pt")
        self.max_vocab_size = max_vocab_size

        self.min_freq = min_freq
        self.specials = specials
        # self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"context": TextField(tokenize_fn=None, special_tokens=self.specials),
                      "response": TextField(tokenize_fn=None, special_tokens=self.specials)}

        if share_vocab:
            self.field["response"] = self.field["context"]

        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            logger.info("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field()

    def build(self):
        # data_context_file = os.path.join(self.args.data_dir, self.args.context_file)
        data_response_file = os.path.join(self.args.data_dir, self.args.response_file)

        logger.info("Reading Data ...")
        train_examples = self.read_and_build_examples(data_response_file, data_type="train")
        valid_examples = self.read_and_build_examples(data_response_file, data_type="valid")
        test_examples = self.read_and_build_examples(data_response_file, data_type="test")

        # 根据训练集来定制词表
        logger.info("Build Vocab ...")
        self.build_vocab(train_examples)

        self.data = {"train": train_examples,
                     "valid": valid_examples,
                     "test": test_examples
                     }

        logger.info("Saved context field to '{}'".format(self.field_context_file))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        field_context = {"itos": self.field["context"].itos,
                      "stoi": self.field["context"].stoi,
                      "vocab_size": self.field["context"].vocab_size,
                      "specials": self.field["context"].specials
                      }
        torch.save(field_context, self.field_context_file)

        logger.info("Saved response field to '{}'".format(self.field_response_file))
        field_response = {"itos": self.field["response"].itos,
                      "stoi": self.field["response"].stoi,
                      "vocab_size": self.field["response"].vocab_size,
                      "specials": self.field["response"].specials
                      }
        torch.save(field_response, self.field_response_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    @timer
    def read_and_build_examples(self, data_response_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 question, answer
        """
        if data_type == "train":
            data_context_file = os.path.join(self.args.data_dir, self.args.train_file)
        elif data_type == "valid":
            data_context_file = os.path.join(self.args.data_dir, self.args.dev_file)
        elif data_type == "test":
            data_context_file = os.path.join(self.args.data_dir, self.args.test_file)
        else:
            data_context_file = None

        if not os.path.isfile(data_context_file) or not os.path.isfile(data_response_file):
            logger.info("{} data context and responses can't find in {} or {}".format(
                data_type, data_context_file, data_response_file))
            raise

        examples, len_seq_utterence, len_seq_response = [], [], []
        desc_message = "GET DATA FROM " + data_type.upper()

        f_context = open(data_context_file, 'r', encoding="utf-8", errors="ignore")
        response_dict = self.get_response(data_response_file)

        for line in tqdm(f_context, desc=desc_message):
            # FIXME 全量数据
            # if i % 10000 == 0:
            #     logger.info("Read {} examples from {}".format(len(examples), data_type.upper()))
            # if len(lines) >= 20000:
            #     break
            fields = line.strip().split("\t")
            us_id = fields[0]
            context = fields[1]
            utterances = context.split(" __EOS__ ")
            # new_utterances = []
            # for utterance in utterances:
            #     new_utterances.append(utterance + " __EOS__")
            # utterances = new_utterances[-self.args.max_utter_num:]  # select the last max_utter_num utterances

            us_tokens = []
            for utterance in utterances:
                u_tokens = utterance.split(' ')
                us_tokens.append(u_tokens)
                len_seq_utterence.append(len(u_tokens))

            if fields[3] != "NA":
                neg_ids = [id for id in fields[3].split('|')]
                for r_id in neg_ids:
                    response, r_len = response_dict[r_id]
                    len_seq_response.append(r_len)
                    examples.append(MTRSExample(
                        guid=us_id,
                        utterences=utterances,
                        response=response,
                        label=0.0
                    ))
                    # break  # uncomment this line when testing recall_2@1

            if fields[2] != "NA":
                pos_ids = [id for id in fields[2].split('|')]
                for r_id in pos_ids:
                    response, r_len = response_dict[r_id]
                    len_seq_response.append(r_len)
                    examples.append(MTRSExample(
                        guid=us_id,
                        utterences=utterances,
                        response=response,
                        label=1.0
                    ))

        len_seq_utterence = np.array(len_seq_utterence)
        len_seq_response = np.array(len_seq_response)
        logger.info("utterences {} sequence length converge 95%".format(np.percentile(len_seq_utterence, 95)))
        logger.info("response {} sequence length converge 95%".format(np.percentile(len_seq_response, 95)))
        return examples

    def get_response(self, data_response_file):
        responses = {}
        with open(data_response_file, 'r', encoding="utf-8", errors="ignore") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) != 2:
                    print("WRONG LINE: {}".format(line))
                    response = 'unknown'
                else:
                    response = fields[1]
                tokens = response.split(' ')
                responses[fields[0]] = (response, len(tokens))
        return responses

    def build_vocab(self, examples):
        """
        从train的text分别生成字典
        data format [{"question":, "answer":},...]
        """
        vocab_file = os.path.join(self.args.data_dir, self.args.vocab_file)
        if os.path.isfile(vocab_file):
            self.load_vocab(vocab_file)
        else:
            xs = [[example.utterences, example.response] for example in examples]
            self.field["context"].build_vocab(xs,
                                              min_freq=self.min_freq,
                                              max_size=self.max_vocab_size)

    def load_vocab(self, file_name):
        with open(file_name, 'r', encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split('\t')
                self.field["context"].stoi[fields[0]] = int(fields[1])
        self.field["context"].itos = [0] * (len(self.field["context"].stoi)+1)
        self.field["context"].itos[0] = constants.PAD
        for key, val in self.field["context"].stoi.items():
            self.field["context"].itos[val] = key

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        logger.info("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)

    def load_field(self):
        context_field = torch.load(self.field_context_file)
        response_field = torch.load(self.field_response_file)
        self.field["context"].load(context_field)
        self.field["response"].load(response_field)

    def create_batch(self, data_type="train"):
        examples = self.data[data_type]
        # FIXME Check example num
        # examples = examples[0:1024]
        features_cache_path = os.path.join(
            self.args.output_dir,
            "features-{}-{}-{}.pt".format(data_type, self.args.max_seq_length, self.args.max_utter_num)
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert {} examples to features".format(data_type))
            features = self.convert_examples_to_features(examples, data_type)
            torch.save(features, features_cache_path)
        # 按需修改
        all_utters_id = torch.tensor([f.utters_id for f in features], dtype=torch.long)
        all_utters_len = torch.tensor([f.utters_len for f in features], dtype=torch.long)
        all_utters_num = torch.tensor([f.utters_num for f in features], dtype=torch.long)
        all_response_id = torch.tensor([f.response_id for f in features], dtype=torch.long)
        all_response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_utters_id, all_utters_len, all_utters_num, all_response_id, all_response_len, all_label)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def convert_examples_to_features(self, examples, data_type="train"):
        features = []
        desc_message = "GET Feature FROM " + data_type.upper()
        for example in tqdm(examples, desc=desc_message):
            utterances = example.utterences[-self.args.max_utter_num:]
            us_tokens, us_vec, us_len = [], [], []
            for utterance in utterances:
                u_tokens = utterance.split(' ')[
                           :self.args.max_seq_length]  # select the first max_utter_len tokens in every utterance
                u_len, u_vec = self.str2num(u_tokens, field_type="context")
                us_tokens.append(u_tokens)
                us_vec.append(u_vec)
                us_len.append(u_len)
            us_num = len(utterances)
            us_vec, us_len = self.padding_context(us_vec, us_len)

            r_tokens = example.response.split(" ")[:self.args.max_seq_length]
            r_len, r_vec = self.str2num(r_tokens, field_type="response")
            r_vec = self.padding_seq(r_vec)

            features.append(MTRSFeatures(
                utters_id=us_vec,
                utters_len=us_len,
                utters_num=us_num,
                response_id=r_vec,
                response_len=r_len,
                label=example.label
            ))
        return features

    def str2num(self, tokens, field_type="context"):
        length = len(tokens)
        vec = []
        for token in tokens:
            vec.append(self.field[field_type].stoi.get(token, self.field[field_type].stoi.get("unknown")))
        return length, np.array(vec)

    def padding_seq(self, seq):
        if len(seq) == self.args.max_seq_length:
            return seq
        new_vec = np.zeros(self.args.max_seq_length, dtype='int32')
        for i in range(len(seq)):
            new_vec[i] = seq[i]
        return new_vec

    def padding_context(self, us_vec, us_len):
        new_utters_vec = np.zeros((self.args.max_utter_num, self.args.max_seq_length), dtype='int32')
        new_utters_len = np.zeros((self.args.max_utter_num,), dtype='int32')
        for i in range(len(us_len)):
            new_utter_vec = self.padding_seq(us_vec[i])
            new_utters_vec[i] = new_utter_vec
            new_utters_len[i] = us_len[i]
        return new_utters_vec, new_utters_len

    # def padding_char_seq(self, seq, max_len, pad_id, max_char_len):
    #     for i in range(len(seq)):
    #         padding_char_length = max_char_len - len(seq[i])
    #         seq[i] += [pad_id] * padding_char_length
    #     padding_length = max_len - len(seq)
    #     seq += [[pad_id for i in range(max_char_len)]] * padding_length
    #     return seq
    #
    # def padding_char_len(self, seq, max_len, pad_id):
    #     padding_length = max_len - len(seq)
    #     seq += [pad_id] * padding_length
    #     return seq


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/gong/NLPData/MTRS/Ubuntu_V1",
        type=str,
    )
    parser.add_argument(
        "--train_file",
        default="train.txt",
        type=str,
    )
    parser.add_argument(
        "--dev_file",
        default="valid.txt",
        type=str,
    )
    parser.add_argument(
        "--test_file",
        default="test.txt",
        type=str,
    )
    parser.add_argument(
        "--response_file",
        default="responses.txt",
        type=str,
    )
    parser.add_argument(
        "--vocab_file",
        default="vocab.txt",
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--max_utter_num",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--train_batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default="/home/gong/zz/data/ubuntu_v1/",
        type=str,
    )
    parser.add_argument("--aug", action="store_true")

    args, _ = parser.parse_known_args()
    # print(args)
    processor = MTRSCorpus(args)
    # train_iter = processor.create_batch("train")
    valid_iter = processor.create_batch("valid")
    test_iter = processor.create_batch("test")
