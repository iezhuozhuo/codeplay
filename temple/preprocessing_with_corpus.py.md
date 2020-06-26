### corpus.py的相关模板，可供参考规范代码


- 使用corpus()作为父类，用于读取并处理数据。
- Field()类用于完成pretrain Embedding的读取以及vocab表的构建。


corpus类普通使用方法：

```python
import os
from tqdm import tqdm

import torch

from source.inputters.field import TextField
from source.inputters.field import NumberField
from source.inputters.dataset import Dataset
#from source.inputters.corpus import Corpus

class classiflyCorpus(object):
    """
    Corpus
    """

    def __init__(self, args,
                 use_word=False,
                 max_vocab_size=10000, min_freq=1, sort_fn =None,
                 embed_file=None):
        self.args = args
        self.data_dir = self.args.data_dir
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = 'classify' + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = 'classify' + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(self.data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(self.data_dir, prepared_vocab_file)

        self.filter_pred = None
        self.sort_fn = sort_fn
        self.data = None

        self.tokenizer = self.get_tokenizer(use_word=use_word)
        self.SRC = TextField(tokenize_fn=self.tokenizer,
                             embed_file=embed_file)

        self.TGT = NumberField()
        self.fields = {'src': self.SRC, 'tgt': self.TGT}
        self.embed_file = embed_file

    def load(self):
        """
        加载.pt文件
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.SRC.stoi[self.SRC.pad_token]

    def reload(self, datafile, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, datafile)
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        加载.pt格式的语料
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        加载.pt格式的字典
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size)
                       for name, field in self.fields.items()
                       if isinstance(field, TextField)))

    def read_data(self, data_file, data_type="train"):
        """
        读取样本文件

        Return:
            data: 字典列表，每个字典由src, tgt, cue组成，cue是个知识列表
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                src, tgt = lin.split('\t')
                data.append({'src': src, 'tgt': tgt})

        #print("Read {} {} examples ({} filtered)".format(len(data), data_type.upper()))
        return data

    def build_vocab(self, data):
        """
        从样本的src分别生成字典
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data):
        """
        将样本的src, tgt两个部分分别索引化
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = self.fields[name].numericalize(strings)
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        加载样本并从样本生成字典和索引形式的语料
        其中，字典是只通过训练集生成的
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.args.train_file)
        valid_file = os.path.join(self.data_dir, self.args.dev_file)
        test_file = os.path.join(self.data_dir, self.args.test_file)

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")
        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader

    def get_tokenizer(self, use_word=False):
        if use_word:
            tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        else:
            tokenizer = lambda x: [y for y in x]  # char-level
        return tokenizer
```

- 使用field类用于数据域的定义，其存在于source.inputters.field包中。其父类如下：

```python
class Field(object):
    """
    Field
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings):
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings)
        else:
            return [self.numericalize(s) for s in strings]

    def denumericalize(self, numbers):
        """
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
```

- 将field设计为两种模式，一种为NumberField，用于处理数字数据；一种为TextField，用于处理文本类数据。

```python
class NumberField(Field):
    """
    NumberField
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        super(NumberField, self).__init__(sequential=sequential,
                                          dtype=dtype)

    def str2num(self, string):
        """
        str2num
        """
        if self.sequential:
            return [self.dtype(s) for s in string.split(" ")]
        else:
            return self.dtype(string)

    def num2str(self, number):
        """
        num2str
        """
        if self.sequential:
            return " ".join([str(x) for x in number])
        else:
            return str(number)


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 tokenize_fn=None,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 special_tokens=None,
                 embed_file=None):
        super(TextField, self).__init__(sequential=True,
                                        dtype=int)
        self.tokenize_fn = tokenize_fn if tokenize_fn is not None else str.split
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.embed_file = embed_file

        specials = [self.pad_token, self.unk_token,
                    self.bos_token, self.eos_token]
        self.specials = [x for x in specials if x is not None]

        if special_tokens is not None:
            for token in special_tokens:
                if token not in self.specials:
                    self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, min_freq=0, max_size=None):
        """
        build_vocab
        """
        def flatten(xs):
            """
            flatten
            """
            flat_xs = []
            for x in xs:
                if isinstance(x, str):
                    flat_xs.append(x)
                elif isinstance(x[0], str):
                    flat_xs += x
                else:
                    flat_xs += flatten(x)
            return flat_xs

        # flatten texts
        texts = flatten(texts)

        counter = Counter()
        for string in tqdm(texts):
            tokens = self.tokenize_fn(string)
            counter.update(tokens)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        self.itos = list(self.specials)

        if max_size is not None:
            max_size = max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
            cover += freq
        cover = cover / sum(freq for _, freq in words_and_frequencies)
        print(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover))

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file)

    def build_word_embeddings(self, embed_file):
        """
        build_word_embeddings
        """

        filename_trimmed_dir = os.path.join('.//output//', "embedding_SougouNews.npz")

        if os.path.exists(filename_trimmed_dir):
            embeddings = np.load(filename_trimmed_dir)["embeddings"].astype('float32')
            return embeddings

        embeddings = np.random.rand(len(self.stoi), 300)
        with open(embed_file, "r", encoding='UTF-8') as f:
            for i, line in enumerate(f.readlines()):
                lin = line.strip().split(" ")
                if lin[0] in self.stoi:
                    idx = self.stoi[lin[0]]
                    emb = [float(x) for x in lin[1:301]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
            np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
        return embeddings


    def dump_vocab(self):
        """
        dump_vocab
        """
        vocab = {"itos": self.itos,
                 "embeddings": self.embeddings}
        return vocab

    def load_vocab(self, vocab):
        """
        load_vocab
        """
        self.itos = vocab["itos"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.embeddings = vocab["embeddings"]

    def str2num(self, string):
        """
        str2num
        """
        tokens = []
        unk_idx = self.stoi[self.unk_token]

        if self.bos_token:
            tokens.append(self.bos_token)

        tokens += self.tokenize_fn(string)

        if self.eos_token:
            tokens.append(self.eos_token)
        indices = [self.stoi.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text
```