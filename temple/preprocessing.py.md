### preprocessing.py的相关模板，可供参考规范代码

两种方式：

- 使用DataFiledProcessor()类，用的是torchtext封装[目前暂时没有]
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


class DataProcessor():
    def __init__(self, args, use_word=False, max_size=10000, min_freq=1):
        super(DataProcessor, self).__init__()
        self.args = args
        self.tokenizer = self.get_tokenizer(use_word=use_word)
        self.vocab = self.get_vocab_dic(max_size=max_size, min_freq=min_freq)

    def get_dataset(self, file_path):
        # 各种读入数据的处理，包括word2id、padding_index
        examples = []
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
                examples.append((words_line, int(label), seq_len)) # [([...], 0), ([...], 1), ...]
        # examples = examples[0:1024]
        all_inputs_id = torch.tensor([f[0] for f in examples], dtype=torch.long)
        all_inputs_label = torch.tensor([f[1] for f in examples], dtype=torch.long)
        all_inputs_len = torch.tensor([f[2] for f in examples], dtype=torch.long)
        dataset = TensorDataset(all_inputs_id, all_inputs_label, all_inputs_len)
        return dataset

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
        # 按要求构建vocab
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
        # 此处定制个人的tokenizer处理器
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        return tokenizer

    def get_vocab_dic(self, max_size=10000, min_freq=1):
        # 读vocab或者构建vocab并保存
        if os.path.exists(self.args.vocab_path):
            vocab = json.load(open(self.args.vocab_path, 'r', encoding="utf-8"))
        else:
            train_file_path = os.path.join(self.args.data_dir, self.args.train_file)
            vocab = self.build_vocab(train_file_path,
                                     max_size=max_size, min_freq=min_freq
                                     )
            json.dump(vocab, open(self.args.vocab_path, 'w', encoding="utf-8"))
        return vocab
```

HuggingFace数据处理方式：【待更新】

DataFiledProcessor()类处理方式：【待更新】

