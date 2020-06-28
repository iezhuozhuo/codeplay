import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import source.utils.Constant as constants
from source.inputters.field import TextField, NumberField


class CNERCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None):
        super(CNERCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.output_dir, "data.pt")
        self.field_text_file = os.path.join(args.output_dir, "field_text.pt")
        self.field_label_file = os.path.join(args.output_dir, "field_label.pt")
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.specials = specials
        self.label_list = self.get_labels()
        self.tokenizer = self.get_tokenizer()

        print("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=self.tokenizer, special_tokens=self.specials),
                      "label": NumberField(label_list=self.label_list)}
        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            print("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field()

    def build(self):
        train_file = os.path.join(self.args.data_dir, self.args.train_file)
        valid_file = os.path.join(self.args.data_dir, self.args.dev_file)
        test_file = os.path.join(self.args.data_dir, self.args.test_file)

        print("Reading Data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        print("Build Vocab from {} ...".format(train_file))
        # 根据训练集来定制词表
        self.build_vocab(train_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     "test": test_data}

        print("Saved text field to '{}'".format(self.field_text_file))
        field_text = {"itos": self.field["text"].itos,
                 "stoi": self.field["text"].stoi,
                 "vocab_size": self.field["text"].vocab_size,
                 "specials": self.field["text"].specials
                 }
        torch.save(field_text, self.field_text_file)

        print("Saved label field to '{}'".format(self.field_label_file))
        field_label = {"label_list": self.field["label"].label_list,
                       "id2label": self.field["label"].id2label,
                       "label2id": self.field["label"].label2id}
        torch.save(field_label, self.field_label_file)

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
        examples, len_seq = [], []
        desc_message = "GETDATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            words = self.tokenizer(data["text"])
            len_seq.append(len(words))
            tokens, labels = [], []

            for i, word in enumerate(words):
                tokens.append(self.field["text"].stoi.get(word, self.field["text"].stoi.get(constants.UNK_WORD)))
                labels.append(self.field["label"].str2num(self.prune_label(data["labels"][i])))

            special_tokens_count = 2
            if len(tokens) > self.args.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.args.max_seq_length - special_tokens_count)]
                labels = labels[: (self.args.max_seq_length - special_tokens_count)]

            tokens += [self.field["text"].stoi[constants.SEP]]
            labels += [self.field["label"].str2num(constants.SEP)]
            tokens = [self.field["text"].stoi[constants.CLS]] + tokens
            labels = [self.field["label"].str2num(constants.CLS)] + labels

            seq_len = len(labels)

            padding_length = self.args.max_seq_length - len(tokens)
            tokens += [self.field["text"].stoi[constants.PAD_WORD]] * padding_length
            labels += [self.field["text"].stoi[constants.PAD_WORD]] * padding_length

            examples.append((tokens, labels, seq_len))
        len_seq = np.array(len_seq)
        print("{} sequence length converge 95%".format(np.percentile(len_seq, 95)))
        return examples

    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由text, label
        """
        lines = []
        with open(data_file, 'r', encoding="utf-8") as f:
            words, labels = [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"text": " ".join(words), "labels": labels})
                        words, labels = [], []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"text": " ".join(words), "labels": labels})
        print("Read {} examples from {})".format(len(lines), data_type.upper()))
        return lines

    def get_tokenizer(self):
        return str.split

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self):
        text_field = torch.load(self.field_text_file)
        self.field["text"].load(text_field)

        label_field = torch.load(self.field_label_file)
        self.field["label"].load(label_field)

    def create_batch(self, data_type="train"):
        # TODO check example num
        examples = self.data[data_type][0:1024]
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

    def get_labels(self):
        """See base class."""
        return ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', constants.CLS, constants.SEP]

    def prune_label(self, x):
        if 'M-' in x:
            return x.replace('M-', 'I-')
        elif 'E-' in x:
            return x.replace('E-', 'I-')
        else:
            return x


if __name__ == "__main__":
    pass

