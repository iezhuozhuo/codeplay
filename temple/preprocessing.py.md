### preprocessing.py的相关模板，可供参考规范代码

两种方式：

- 使用ModelCorpus()类，非预训练读入方式。
- 使用DataProcessor()类，可以和普通的读入方式一样也可以兼容huggingface的处理（产出example便于生成feature）

DataProcessor类的普通使用方法：

```python
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

class ModelCorpus(object):
    def __init__(self,
                 args,
                 use_word=False,#中文
                 max_vocab_size=10000,
                 min_freq=1,
                 specials=None):
        super(ModelCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.output_dir, "data.pt")
        self.field_file = os.path.join(args.output_dir, "field.pt")
        self.use_word = use_word
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.specials = specials

        self.tokenizer = self.get_tokenizer(self.use_word)

        print("Initial Corpus ...")
        self.field = {"text": TextField(tokenize_fn=self.tokenizer, special_tokens=self.specials),
                      "label": NumberField()} # 定制field
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
        # 根据训练集来定制词表
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
        # 定制build_example
        examples = []
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

            examples.append((words_line, int(data["label"]), seq_len))  # [([...], 0), ([...], 1), ...]
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
        # 定制tokenizer
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
        # 定制其他读入
        dataset = TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

```

HuggingFace数据处理方式：【待更新】

