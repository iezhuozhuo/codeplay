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


class Example(object):
    def __init__(self, us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, bi, us_tokens, r_tokens):
        self.us_id = us_id
        self.us_len = us_len
        self.us_vec = us_vec,
        self.us_num = us_num,
        self.r_id = r_id
        self.r_len = r_len
        self.r_vec = r_vec
        self.bi = bi
        self.us_tokens = us_tokens
        self.r_tokens = r_tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 x_utterances, x_response, x_utterances_len, x_response_len, x_utterances_num, targets, target_weights, id_pairs, x_utterances_char, x_utterances_char_len, x_response_char, x_response_char_len):
        self.x_utterances = x_utterances
        self.x_response = x_response
        self.x_utterances_len = x_utterances_len
        self.x_response_len = x_response_len
        self.x_utterances_num = x_utterances_num
        self.targets = targets
        self.target_weights = target_weights
        self.id_pairs = id_pairs
        self.x_utterances_char = x_utterances_char
        self.x_utterances_char_len = x_utterances_char_len
        self.x_response_char = x_response_char
        self.x_response_char_len = x_response_char_len



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
                      "char": TextField(tokenize_fn=None, special_tokens=self.specials),
                     }

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

        vocab, charVocab, vocab_list, charVocab_list = self.build_vocab()




        logger.info("Reading Data ...")
        response_raw = self.load_responses(self.args.response_file, vocab, self.args.max_response_len)

        train_raw = self.read_data(data_train_file, vocab, response_raw, data_type="train")
        valid_raw = self.read_data(data_dev_file, vocab, response_raw, data_type="valid")

        logger.info("Build Vocab from {} ...".format(data_train_file))


        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        # test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     # "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_text_file))
        field_article = {"itos": self.field["text"].itos,
                         "stoi": self.field["text"].stoi,
                         "vocab_size": self.field["text"].vocab_size,
                         "specials": self.field["text"].specials
                         }
        logger.info("Saved text field to '{}'".format(self.field_char_file))
        field_char_article = {"itos": self.field["char"].itos,
                            "stoi": self.field["char"].stoi,
                            "vocab_size": self.field["char"].vocab_size,
                            "specials": self.field["char"].specials
                            }

        torch.save(field_article, self.field_text_file)
        torch.save(field_char_article, self.field_char_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    def load_responses(self, fname, vocab, maxlen):
        responses = {}
        with open(fname, 'rt') as f:
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                if len(fields) != 2:
                    print("WRONG LINE: {}".format(line))
                    r_text = 'unknown'
                else:
                    r_text = fields[1]
                tokens = r_text.split(' ')[:maxlen]
                len1, vec = self.to_vec(tokens, vocab, maxlen)
                responses[fields[0]] = (len1, vec, tokens)
        return responses

    def to_vec(self, tokens, vocab, maxlen):
        '''
        length: length of the input sequence
        vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
        '''
        n = len(tokens)
        length = 0
        vec = []
        for i in range(n):
            length += 1
            if tokens[i] in vocab:
                vec.append(vocab[tokens[i]])
            else:
                vec.append(vocab["unknown"])

        return length, np.array(vec)

    @timer
    def read_data(self, data_file, vocab, responses, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 text_a, text_b, label
        """
        dataset = []
        with open(data_file, 'rt') as f:
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                us_id = fields[0]

                context = fields[1]
                utterances = context.split(" __EOS__ ")
                new_utterances = []
                for utterance in utterances:
                    new_utterances.append(utterance + " __EOS__")
                utterances = new_utterances[-self.args.max_utter_num:]  # select the last max_utter_num utterances

                us_tokens = []
                us_vec = []
                us_len = []
                for utterance in utterances:
                    u_tokens = utterance.split(' ')[
                               :self.args.max_utter_len]  # select the first max_utter_len tokens in every utterance
                    u_len, u_vec = self.to_vec(u_tokens, vocab, self.args.max_utter_len)
                    us_tokens.append(u_tokens)
                    us_vec.append(u_vec)
                    us_len.append(u_len)

                us_num = len(utterances)

                if fields[2] != "NA":
                    pos_ids = [id for id in fields[2].split('|')]
                    for r_id in pos_ids:
                        r_len, r_vec, r_tokens = responses[r_id]
                        dataset.append((us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, 1.0, us_tokens, r_tokens))

                if fields[3] != "NA":
                    neg_ids = [id for id in fields[3].split('|')]
                    for r_id in neg_ids:
                        r_len, r_vec, r_tokens = responses[r_id]
                        dataset.append((us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, 0.0, us_tokens, r_tokens))
        logger.info("Read total {} examples from {}".format(len(dataset), data_type.upper()))
        return dataset

    def build_vocab(self):
        """
        从train的text分别生成字典
        data format [{"text_a":, "text_b":},...]
        """
        vocab, vocab_list = self.load_vocab(self.args.vocab_file)
        charVocab, charVocab_list = self.load_char_vocab(self.args.char_vocab_file)


        return vocab, charVocab, vocab_list, charVocab_list

        # self.field["text"].build_vocab(xs,
        #                                min_freq=self.min_freq,
        #                                max_size=self.max_vocab_size)
        #
        #
        # self.field["char"].build_vocab(cs,
        #                                min_freq=self.min_freq,
        #                                max_size=self.max_char_size)

    def load_vocab(self, fname):
        '''
        vocab = {"<PAD>": 0, ...}
        '''
        vocab = {"<PAD>": 0}
        vocab_list = ["<PAD>"]
        with open(fname, 'rt') as f:
            for line in f:
                # fields = line.decode('utf-8').strip().split('\t')
                fields = line.strip().split('\t')
                vocab[fields[0]] = int(fields[1])
                vocab_list.append(fields[0])

        self.field['text'].itos = vocab_list
        self.field['text'].stoi = vocab
        self.field['text'].vocab_size = len(vocab_list)
        self.field['text'].specials = ["<PAD>"]
        return vocab, vocab_list

    def load_char_vocab(self, fname):
        '''
        charVocab = {"U": 0, "!": 1, ...}
        '''
        charVocab = {}
        charVocab_list = []
        with open(fname, 'rt') as f:
            for line in f:
                fields = line.strip().split('\t')
                char_id = int(fields[0])
                ch = fields[1]
                charVocab[ch] = char_id
                charVocab_list.append(ch)
        self.field['char'].itos = charVocab_list
        self.field['char'].stoi = charVocab
        self.field['char'].vocab_size = len(charVocab_list)
        self.field['char'].specials = []
        return charVocab, charVocab_list


    def build_examples(self, data_raw, data_type="train"):
        if data_raw == None:
            logger.info("{} data text and label can't find".format(data_type))
        examples = []
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            examples.append(Example(
                    us_id = data[0],
                    us_len = data[1],
                    us_vec = data[2],
                    us_num = data[3],
                    r_id = data[4],
                    r_len = data[5],
                    r_vec = data[6],
                    bi = data[7],
                    us_tokens = data[8],
                    r_tokens = data[9]
            ))
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
            "features-{}-{}-{}".format(data_type, self.args.max_seq_length, "aug" if self.args.aug else "no-aug")
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading prepared features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert examples to features")
            features = self.convert_examples_to_features(examples)
            torch.save(features, features_cache_path)

        x_utterances = torch.tensor([f.x_utterances for f in features], dtype=torch.long)
        x_response = torch.tensor([f.x_response for f in features], dtype=torch.long)
        x_utterances_len = torch.tensor([f.x_utterances_len for f in features], dtype=torch.long)
        x_response_len = torch.tensor([f.x_response_len for f in features], dtype=torch.long)
        x_utterances_num = torch.tensor([f.x_utterances_num for f in features], dtype=torch.long)
        targets = torch.tensor([f.targets for f in features], dtype=torch.long)
        x_utterances_char = torch.tensor([f.x_utterances_char for f in features], dtype=torch.long)
        x_utterances_char_len = torch.tensor([f.x_utterances_char_len for f in features], dtype=torch.long)
        x_response_char = torch.tensor([f.x_response_char for f in features], dtype=torch.long)
        x_response_char_len = torch.tensor([f.x_response_char_len for f in features], dtype=torch.long)
        dataset = TensorDataset(x_utterances, x_response, x_utterances_len, x_response_len, x_utterances_num, targets, x_utterances_char, x_utterances_char_len, x_response_char, x_response_char_len)

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
        target_loss_weights = [1.0, 1.0]
        features = []
        desc_message = "GETDATA FROM " + data_type.upper()
        x_utterances = []
        x_response = []
        x_utterances_len = []
        x_response_len = []
        targets = []
        target_weights = []
        id_pairs = []

        x_utterances_char = []
        x_utterances_char_len = []
        x_response_char = []
        x_response_char_len = []

        x_utterances_num = []
        for example in tqdm(examples, desc=desc_message):
            us_id, us_len, us_vec, us_num, r_id, r_len, r_vec, label, us_tokens, r_tokens = example.us_id, example.us_len,example.us_vec,example.us_num,example.r_id,example.r_len,example.r_vec,example.bi,example.us_tokens,example.r_tokens
            if label > 0:
                t_weight = target_loss_weights[1]
                target_weights.append(target_loss_weights[1])
            else:
                t_weight = target_loss_weights[0]
                target_weights.append(target_loss_weights[0])

            # normalize us_vec and us_len
            new_utters_vec = np.zeros((self.args.max_utter_num, self.args.max_utter_len), dtype='int32')
            new_utters_len = np.zeros((self.args.max_utter_num,), dtype='int32')
            # new_utters_vec = [[0]*self.args.max_utter_len for _ in range(self.args.max_utter_num)]
            # new_utters_len = [0] * self.args.max_utter_num


            for i in range(len(us_len)):
                new_utter_vec = self.normalize_vec(us_vec[0][i], self.args.max_utter_len)
                new_utters_vec[i] = new_utter_vec
                new_utters_len[i] = us_len[i]
            new_r_vec = self.normalize_vec(r_vec, self.args.max_response_len)

            x_utterances.append(new_utters_vec)
            x_utterances_len.append(new_utters_len)
            x_response.append(new_r_vec)
            x_response_len.append(r_len)
            targets.append(label)
            id_pairs.append((us_id, r_id, int(label)))

            # normalize CharVec and CharLen
            uttersCharVec = np.zeros((self.args.max_utter_num, self.args.max_utter_len, self.args.maxWordLength), dtype='int32')
            uttersCharLen = np.ones((self.args.max_utter_num, self.args.max_utter_len), dtype='int32')
            for i in range(len(us_len)):
                utterCharVec, utterCharLen = self.charVec(us_tokens[i], self.field['char'].stoi, self.args.max_utter_len, self.args.maxWordLength)
                uttersCharVec[i] = utterCharVec
                uttersCharLen[i] = utterCharLen
            rCharVec, rCharLen = self.charVec(r_tokens, self.field['char'].stoi, self.args.max_response_len, self.args.maxWordLength)

            x_utterances_char.append(uttersCharVec)
            x_utterances_char_len.append(uttersCharLen)
            x_response_char.append(rCharVec)
            x_response_char_len.append(rCharLen)

            x_utterances_num.append(us_num)

            # assert len(left_ids) == self.args.max_seq_length
            # assert len(right_ids) == self.args.max_seq_length
            # assert len(left_char_ids) == self.args.max_seq_length
            # assert len(right_char_ids) == self.args.max_seq_length
            features.append(InputFeatures(
                            x_utterances=new_utters_vec.tolist(),
                            x_response=new_r_vec.tolist(),
                            x_utterances_len=new_utters_len.tolist(),
                            x_response_len=[r_len],
                            x_utterances_num=[us_num],
                            targets=[label],
                            target_weights=[t_weight],
                            id_pairs=(us_id, r_id, int(label)),
                            x_utterances_char=uttersCharVec.tolist(),
                            x_utterances_char_len=uttersCharLen.tolist(),
                            x_response_char=rCharVec.tolist(),
                            x_response_char_len=rCharLen.tolist()
            ))


        x_response_char_len = np.array(x_response_char_len)
        x_utterances_char_len = np.array(x_utterances_char_len)
        x_utterances_len = np.array(x_utterances_len)
        x_response_len = np.array(x_response_len)
        print("{} sequence length converge 95%".format(np.percentile(x_response_char_len, 95)))
        print("{} sequence length converge 95%".format(np.percentile(x_utterances_char_len, 95)))
        print("{} char length converge 95%".format(np.percentile(x_utterances_len, 95)))
        print("{} char length converge 95%".format(np.percentile(x_response_len, 95)))
        return features


    def charVec(self, tokens, charVocab, maxlen, maxWordLength):
        '''
        chars = np.array( (maxlen, maxWordLength) )    0 if not found in charVocab or None
        word_lengths = np.array( maxlen )              1 if None
        '''
        n = len(tokens)
        if n > maxlen:
            n = maxlen

        chars = np.zeros((maxlen, maxWordLength), dtype=np.int32)
        word_lengths = np.ones(maxlen, dtype=np.int32)
        for i in range(n):
            token = tokens[i][:maxWordLength]
            word_lengths[i] = len(token)
            row = chars[i]
            for idx, ch in enumerate(token):
                if ch in charVocab:
                    row[idx] = charVocab[ch]

        return chars, word_lengths

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

    def normalize_vec(self, vec, maxlen):
        '''
        pad the original vec to the same maxlen
        [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
        '''
        if len(vec) == maxlen:
            return vec

        new_vec = np.zeros(maxlen, dtype='int32')
        for i in range(len(vec)):
            new_vec[i] = vec[i]
        return new_vec

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