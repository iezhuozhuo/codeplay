import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.data import Field, TabularDataset

import source.utils.Constant as constants
from source.inputters.corpus import Corpus
from source.inputters.field import TextField, NumberField


class DataProcessor():
    def __init__(self, args, use_word=False, max_size=10000, min_freq=1):
        super(DataProcessor, self).__init__()
        self.args = args
        self.tokenizer = self.get_tokenizer(use_word=use_word)
        self.vocab = self.get_vocab_dic(max_size=max_size, min_freq=min_freq)

    def get_dataset(self, file_path):
        examples = []
        buckets = None
        n_gram = ()
        if hasattr(self.args, "n_gram_vocab"):
            buckets = self.args.n_gram_vocab
        desc_message = "GETDATA FROM" + file_path.upper()
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc=desc_message):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = self.tokenizer(content)
                seq_len = len(token)

                if len(token) < self.args.max_seq_length:
                    token.extend([constants.PAD_WORD] * (self.args.max_seq_length - len(token)))
                else:
                    token = token[:self.args.max_seq_length]
                    seq_len = self.args.max_seq_length
                # word to id
                for word in token:
                    words_line.append(self.vocab.get(word, self.vocab.get(constants.UNK_WORD)))

                # fasttext
                if buckets:
                    bigram = []
                    trigram = []
                    for i in range(self.args.max_seq_length):
                        bigram.append(self.biGramHash(words_line, i, buckets))
                        trigram.append(self.triGramHash(words_line, i, buckets))
                    n_gram = (bigram, trigram)
                examples.append((words_line, int(label), seq_len,) + n_gram) # [([...], 0), ([...], 1), ...]
        # TODO example nums check
        examples = examples[0:1024]
        all_inputs_id = torch.tensor([f[0] for f in examples], dtype=torch.long)
        all_inputs_label = torch.tensor([f[1] for f in examples], dtype=torch.long)
        all_inputs_len = torch.tensor([f[2] for f in examples], dtype=torch.long)
        if buckets:
            all_inputs_bigram = torch.tensor([f[3] for f in examples], dtype=torch.long)
            all_inputs_trigram = torch.tensor([f[4] for f in examples], dtype=torch.long)
            return TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len, all_inputs_bigram, all_inputs_trigram)

        return TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len)

    def get_train_features(self):
        train_file_path = os.path.join(self.args.data_dir, self.args.train_file)
        train_dataset = self.get_dataset(train_file_path)
        return train_dataset

    def get_dev_features(self):
        dev_file_path = os.path.join(self.args.data_dir, self.args.dev_file)
        dev_dataset = self.get_dataset(dev_file_path)
        return dev_dataset

    def get_test_features(self):
        test_file_path = os.path.join(self.args.data_dir, self.args.test_file)
        test_dataset = self.get_dataset(test_file_path)
        return test_dataset

    def build_vocab(self, file_path, max_size, min_freq):
        vocab_dic = {}
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc="Build_Vocab"):
                if not line:
                    continue
                lin = line.strip()
                content = lin.split('\t')[0]
                for word in self.tokenizer(content):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
            vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1],
                                reverse=True)[:max_size]
            vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
            vocab_dic.update({constants.UNK_WORD: len(vocab_dic), constants.PAD_WORD: len(vocab_dic) + 1})
        return vocab_dic

    def get_tokenizer(self, use_word=False):
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        return tokenizer

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def get_vocab_dic(self, max_size=10000, min_freq=1):
        if os.path.exists(self.args.vocab_path):
            vocab = json.load(open(self.args.vocab_path, 'r', encoding="utf-8"))
        else:
            train_file_path = os.path.join(self.args.data_dir, self.args.train_file)
            vocab = self.build_vocab(train_file_path,
                                     max_size=max_size, min_freq=min_freq
                                     )
            json.dump(vocab, open(self.args.vocab_path, 'w', encoding="utf-8"))
        return vocab


class ClassifierCorpus(object):
    def __init__(self,
                 args,
                 use_word=False,
                 max_vocab_size=10000,
                 min_freq=1,
                 specials=None):
        super(ClassifierCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.output_dir, "data.pt")
        self.field_file = os.path.join(args.output_dir, "field.pt")
        self.use_word = use_word
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.specials = specials
        self.buckets = None

        self.tokenizer = self.get_tokenizer(self.use_word)

        print("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=self.tokenizer, special_tokens=self.specials),
                      "label": NumberField()}
        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            print("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field(self.field_file)

    def build(self):
        train_file = os.path.join(self.args.data_dir, self.args.train_file)
        valid_file = os.path.join(self.args.data_dir, self.args.dev_file)
        test_file = os.path.join(self.args.data_dir, self.args.test_file)

        print("Reading Data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        print("Build Vocab from {} ...".format(train_file))
        self.build_vocab(train_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saved field to '{}'".format(self.field_file))
        field = {"itos": self.field["text"].itos,
                 "stoi": self.field["text"].stoi,
                 "vocab_size": self.field["text"].vocab_size,
                 "specials": self.field["text"].specials
                 }
        torch.save(field, self.field_file)

        print("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"text":, "label":},...]
        """
        xs = [x["text"] for x in data]
        self.field["text"].build_vocab(xs,
                          min_freq=self.min_freq,
                          max_size=self.max_vocab_size)

    def build_examples(self, data_raw, data_type="train"):
        examples = []
        n_gram = ()
        if hasattr(self.args, "n_gram_vocab"):
            self.buckets = self.args.n_gram_vocab
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            token = self.tokenizer(data["text"])
            seq_len = len(token)
            words_line = []
            if len(token) < self.args.max_seq_length:
                token.extend([constants.PAD_WORD] * (self.args.max_seq_length - len(token)))
            else:
                token = token[:self.args.max_seq_length]
                seq_len = self.args.max_seq_length
            # word to id
            for word in token:
                words_line.append(self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))

            # fasttext
            if self.buckets:
                bigram = []
                trigram = []
                for i in range(self.args.max_seq_length):
                    bigram.append(self.biGramHash(words_line, i, self.buckets))
                    trigram.append(self.triGramHash(words_line, i, self.buckets))
                n_gram = (bigram, trigram)
            examples.append((words_line, int(data["label"]), seq_len,) + n_gram)  # [([...], 0), ([...], 1), ...]
        return examples

    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由text, label
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                text, label = lin.split('\t')
                data.append({'text': text, 'label': int(label)})
        print("Read {} examples from {})".format(len(data), data_type.upper()))
        return data

    def get_tokenizer(self, use_word=False):
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        return tokenizer

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self, field_file=None):
        field_file = field_file or self.field_file
        field = torch.load(field_file)
        self.field["text"].load(field)

    def create_batch(self, data_type="train"):
        examples = self.data[data_type][0:1024]
        all_inputs_id = torch.tensor([f[0] for f in examples], dtype=torch.long)
        all_inputs_label = torch.tensor([f[1] for f in examples], dtype=torch.long)
        all_inputs_len = torch.tensor([f[2] for f in examples], dtype=torch.long)
        if self.buckets:
            all_inputs_bigram = torch.tensor([f[3] for f in examples], dtype=torch.long)
            all_inputs_trigram = torch.tensor([f[4] for f in examples], dtype=torch.long)
            dataset = TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len, all_inputs_bigram, all_inputs_trigram)
        else:
            dataset = TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


if __name__ == "__main__":
    pass