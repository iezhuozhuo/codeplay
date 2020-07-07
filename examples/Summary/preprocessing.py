import os
import json
import time
import jieba
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer

from train_utils import init_logger

import source.utils.Constant as constants
from source.inputters.field import TextField, NumberField
from source.inputters.dataset import DataProcessor


logger = init_logger()


def timer(func):
    """耗时装饰器，计算函数运行时长"""
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        cost = end - start
        logger.info(f"Cost time: {cost} s")
        return r

    return wrapper


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 article_ids, article_len, article_mask,
                 summary_input_ids, summary_len, summary_taget_ids, summary_mask,
                 article_ids_extend_vocab=None, article_oovs=None, extra_zeros=None,):
        # normal softmax or ner
        self.article_ids = article_ids
        self.article_len = article_len
        self.article_mask = article_mask
        self.summary_input_ids = summary_input_ids
        self.summary_len = summary_len
        self.summary_taget_ids = summary_taget_ids
        self.summary_mask = summary_mask

        self.article_ids_extend_vocab = article_ids_extend_vocab
        self.article_oovs = article_oovs
        self.extra_zeros = extra_zeros


class SummaGenCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None, share_vocab=True):
        super(SummaGenCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.output_dir, "data.pt")
        self.field_article_file = os.path.join(args.output_dir, "field_article.pt")
        self.field_summary_file = os.path.join(args.output_dir, "field_summary.pt")
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.specials = specials
        # self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"article": TextField(tokenize_fn=None, special_tokens=self.specials),
                      "summary": TextField(tokenize_fn=None, special_tokens=self.specials)}

        if share_vocab:
            self.field["summary"] = self.field["article"]

        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            logger.info("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field()

    def build(self):
        data_article_file = os.path.join(self.args.data_dir, self.args.article_file)
        data_summary_file = os.path.join(self.args.data_dir, self.args.summary_file)

        logger.info("Reading Data ...")
        data_raw = self.read_data(data_article_file, data_summary_file, data_type="train")
        random.shuffle(data_raw)
        train_raw = data_raw[:int(len(data_raw) * 0.95)]
        valid_raw = data_raw[int(len(data_raw) * 0.95):]
        # train_raw = data_raw[:600_000]
        # valid_raw = data_raw[600_000:]
        # train_raw = data_raw[:10_000]
        # valid_raw = data_raw[10_000:]
        # test_raw = train_raw[601_000:]
        logger.info("Build Vocab from {} and {} ...".format(data_article_file, data_summary_file))
        # 根据训练集来定制词表
        self.build_vocab(data_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        # test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     #"test": test_data
                     }

        logger.info("Saved article field to '{}'".format(self.field_article_file))
        field_article = {"itos": self.field["article"].itos,
                      "stoi": self.field["article"].stoi,
                      "vocab_size": self.field["article"].vocab_size,
                      "specials": self.field["article"].specials
                      }
        torch.save(field_article, self.field_article_file)

        logger.info("Saved summary field to '{}'".format(self.field_summary_file))
        field_summary = {"itos": self.field["summary"].itos,
                         "stoi": self.field["summary"].stoi,
                         "vocab_size": self.field["summary"].vocab_size,
                         "specials": self.field["summary"].specials
                         }
        torch.save(field_summary, self.field_summary_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    def read_data(self, data_article_file, data_summary_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 article, summary
        """
        if not os.path.isfile(data_article_file) or not os.path.isfile(data_summary_file):
            logger.info("{} data text and label can't find".format(data_type))
            return None

        f_article = open(data_article_file, 'r', encoding="utf-8")
        f_summary = open(data_summary_file, 'r', encoding="utf-8")
        lines = []
        for i, (article, summary) in enumerate(zip(f_article, f_summary)):
            if i % 10000 == 0:
                logger.info("Read {} examples from {}".format(len(lines), data_type.upper()))
            # if len(lines) >= 20000:
            #     break
            article_tokens = self.tokenizer(article)
            summary_tokens = self.tokenizer(summary)
            lines.append({"article": " ".join(article_tokens), "summary": " ".join(summary_tokens)})
        logger.info("Read total {} examples from {}".format(len(lines), data_type.upper()))
        f_article.close()
        f_summary.close()
        return lines

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"article":, "summary":},...]
        """

        xs = [[x["article"], x["summary"]] for x in data]
        self.field["article"].build_vocab(xs,
                                       min_freq=self.min_freq,
                                       max_size=self.max_vocab_size)

    def build_examples(self, data_raw, data_type="train"):
        if data_raw == None:
            logger.info("{} data text and label can't find".format(data_type))
        examples, len_seq_enc, len_seq_dec, len_article_oov = [], [], [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            article_words = str.split(data["article"])
            summary_words = str.split(data["summary"])
            len_seq_enc.append(len(article_words))
            len_seq_dec.append(len(summary_words))

            article_ids, summary_ids = [], []
            for i, word in enumerate(article_words):
                article_ids.append(self.field["article"].stoi.get(word, self.field["article"].stoi.get(constants.UNK_WORD)))
            for i, word in enumerate(summary_words):
                summary_ids.append(self.field["summary"].stoi.get(word, self.field["summary"].stoi.get(constants.UNK_WORD)))

            # process the encoder inputs
            if len(article_ids) > self.args.max_enc_seq_length:
                article_ids = article_ids[:self.args.max_enc_seq_length]
            article_len = len(article_ids)
            # article_ids += [self.field["article"].stoi[constants.BOS_WORD]]
            # article_ids = [self.field["article"].stoi[constants.EOS_WORD]] + article_ids

            summary_input_ids = [self.field["summary"].stoi[constants.BOS_WORD]] + summary_ids
            summary_taget_ids = summary_ids[:]
            if len(summary_input_ids) > self.args.max_dec_seq_length:
                summary_input_ids = summary_input_ids[: self.args.max_dec_seq_length]  # 无结束标志
                summary_taget_ids = summary_taget_ids[: self.args.max_dec_seq_length]
            else:
                summary_taget_ids.append(self.field["summary"].stoi[constants.EOS_WORD])  # 无截断有结束标志
            assert len(summary_input_ids) == len(summary_taget_ids)
            summary_len = len(summary_input_ids)


            # 如果使用pointer-generator模式, 需要一些额外信息
            # 编码时需要输入原文编码和oov单词的编码
            article_ids_extend_vocab, article_oovs, extra_zeros = None, None, None
            if self.args.pointer_gen:
                # 编码时需要输入原文编码和oov单词的编码
                article_ids_extend_vocab, article_oovs = self.article2ids(article_words)

                # 获取参考摘要的id，其中oov单词由原文中的oov单词编码表示
                summary_ids_extend_vocab = self.abstract2ids(summary_words, article_oovs)
                summary_taget_ids = summary_ids_extend_vocab[:]
                if len(summary_ids_extend_vocab) > self.args.max_dec_seq_length:
                    summary_ids_extend_vocab = summary_ids_extend_vocab[: self.args.max_dec_seq_length]  # 无结束标志
                    summary_taget_ids = summary_taget_ids[: self.args.max_dec_seq_length]
                else:
                    summary_taget_ids.append(self.field["summary"].stoi[constants.BOS_WORD])  # 无截断有结束标志
                len_article_oov.append(len(article_oovs))
                extra_zeros = [0] * self.args.max_oov_len

            article_mask = [1] * article_len
            summary_mask = [1] * summary_len

            # padding
            padding_id = self.field["article"].stoi[constants.PAD_WORD]
            article_ids = self.padding_seq(article_ids, self.args.max_enc_seq_length, padding_id)
            article_mask = self.padding_seq(article_mask, self.args.max_enc_seq_length, padding_id)
            # article_oovs = self.
            summary_input_ids = self.padding_seq(summary_input_ids, self.args.max_dec_seq_length, padding_id)
            summary_taget_ids = self.padding_seq(summary_taget_ids, self.args.max_dec_seq_length, padding_id)
            summary_mask = self.padding_seq(summary_mask, self.args.max_dec_seq_length, padding_id)
            if self.args.pointer_gen:
                article_ids_extend_vocab = self.padding_seq(article_ids_extend_vocab, self.args.max_enc_seq_length, padding_id)
                article_oovs = self.padding_seq(article_oovs, self.args.max_oov_len, padding_id)

            examples.append(InputFeatures(
                article_ids=article_ids,
                article_len=article_len,
                article_mask=article_mask,
                summary_input_ids=summary_input_ids,
                summary_taget_ids=summary_taget_ids,
                summary_len=summary_len,
                summary_mask=summary_mask,
                article_ids_extend_vocab=article_ids_extend_vocab,
                article_oovs=article_oovs,
                extra_zeros=extra_zeros))

        len_seq_enc = np.array(len_seq_enc)
        len_seq_dec = np.array(len_seq_dec)
        len_article_oov = np.array(len_article_oov) if len(len_article_oov) > 0 else None
        logger.info("encoder {} sequence length converge 95%".format(np.percentile(len_seq_enc, 95)))
        logger.info("decoder {} sequence length converge 95%".format(np.percentile(len_seq_dec, 95)))
        logger.info("len_article_oov max is {}".format(len_article_oov.max()))
        return examples

    def article2ids(self, article_words):
        """返回两个列表：将文章的词汇转换为id,包含oov词汇id; oov词汇"""
        ids, oovs = [], []
        unk_id = self.field["article"].stoi[constants.UNK_WORD]
        vocab_size = len(self.field["article"].stoi)
        for word in article_words:
            i = self.field["article"].stoi.get(word, self.field["article"].stoi.get(constants.UNK_WORD))
            if i == unk_id:  # If w is OOV
                if word not in oovs:  # Add to list of OOVs
                    oovs.append(word)
                    oov_num = oovs.index(word)  # This is 0 for the first article OOV, 1 for the second article OOV...
                    ids.append(
                        vocab_size + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs

    def abstract2ids(self, abstract_words, article_oovs):
        ids = []
        unk_id = self.field["article"].stoi[constants.UNK_WORD]
        vocab_size = len(self.field["summary"].stoi)
        for word in abstract_words:
            i = self.field["summary"].stoi.get(word, self.field["summary"].stoi.get(constants.UNK_WORD))
            if i == unk_id:  # If w is an OOV word
                if word in article_oovs:  # If w is an in-article OOV
                    vocab_idx = vocab_size + article_oovs.index(word)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(unk_id)  # Map to the UNK token id
            else:
                ids.append(i)
        return ids

    def padding_seq(self, seq, max_len, pad_id):
        padding_length = max_len - len(seq)
        seq += [pad_id] * padding_length
        return seq

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        logger.info("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        logger.info("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self):
        text_field = torch.load(self.field_text_file)
        self.field["text"].load(text_field)

        label_field = torch.load(self.field_label_file)
        self.field["label"].load(label_field)

    def create_batch(self, data_type="train"):
        # TODO check example num
        # examples = self.data[data_type][0:1024]
        examples = self.data[data_type]
        all_inputs_id = torch.tensor([f[0] for f in examples], dtype=torch.long)
        all_inputs_label = torch.tensor([f[1] for f in examples], dtype=torch.long)
        all_inputs_len = torch.tensor([f[2] for f in examples], dtype=torch.long)
        dataset = TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def tokenizer(self, line):
        # jieba.enable_parallel()
        words = jieba.cut(line.strip())
        word_list = list(words)
        # jieba.disable_parallel()
        return word_list

