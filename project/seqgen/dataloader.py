import torch
import random
import logging
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)

PAD, BOS, EOS, UNK = '<_>', '<bos>', '<eos>', '<unk>'


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt


class Vocab(object):
    def __init__(self, filename, with_SE):
        with open(filename) as f:
            if with_SE:
                self.itos = [PAD, BOS, EOS, UNK] + [token.strip() for token in f.readlines()]
            else:
                self.itos = [PAD, UNK] + [token.strip() for token in f.readlines()]
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        self._size = len(self.stoi)
        self._padding_idx = self.stoi[PAD]
        self._unk_idx = self.stoi[UNK]
        self._start_idx = self.stoi.get(BOS, -1)
        self._end_idx = self.stoi.get(EOS, -1)

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self.itos[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self.stoi.get(x, self.unk_idx)

    @property
    def size(self):
        return self._size

    @property
    def padding_idx(self):
        return self._padding_idx

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx


def ListsToTensor(xs, vocab, with_S=False, with_E=False):
    batch_size = len(xs)
    lens = [len(x) + (1 if with_S else 0) + (1 if with_E else 0) for x in xs]
    mx_len = max(max(lens), 1)
    ys = []
    for i, x in enumerate(xs):
        y = ([vocab.start_idx] if with_S else []) + [vocab.token2idx(w) for w in x] + (
            [vocab.end_idx] if with_E else []) + ([vocab.padding_idx] * (mx_len - lens[i]))
        ys.append(y)

    # lens = torch.LongTensor([ max(1, x) for x in lens])
    data = torch.LongTensor(ys).t_().contiguous()
    return data.cuda()


def batchify(data, vocab_src, vocab_tgt):
    src = ListsToTensor([x[0] for x in data], vocab_src)
    tgt = ListsToTensor([x[1] for x in data], vocab_tgt)
    return src, tgt


class DataLoaders(object):
    # def __init__(self, args, filename, vocab_src, vocab_tgt, batch_size, for_train, with_S=False, with_E=False):
    def __init__(self, for_train):
        self.train = for_train
        # self.args = args
        # self.filename = filename
        # self.vocab_src = vocab_src
        # self.vocab_tgt = vocab_tgt
        # self.batch_size = batch_size
        # self.data = self.get_train_data(self.filename)
        # self.with_S = with_S
        # self.with_E = with_E

    def get_a_data(self, line):
        d = [x.split() for x in line.strip().split('|')]
        skip = not (len(d) == 4)
        for j, i in enumerate(d):
            if not self.train:
                d[j] = i[:30]
                if len(d[j]) == 0:
                    d[j] = [UNK]
            if len(i) == 0 or len(i) > 30:
                skip = True
        if not (skip and self.train):
            return d
        else:
            return None

    def get_train_data(self, filename):
        data = []
        with open(filename, encoding="utf-8", errors='ignore') as fin:
            for i, line in enumerate(fin):
                if i % 100000000 == 0:
                    print("Have processed {} lines to examples".format(i))
                a_data = self.get_a_data(line)
                if a_data:
                    data.append(a_data)
        return data

    # def __iter__(self):
    #     idx = list(range(len(self.data)))
    #     if self.train:
    #         random.shuffle(idx)
    #     cur = 0
    #     while cur < len(idx):
    #         data = [self.data[i] for i in idx[cur:cur + self.batch_size]]
    #         cur += self.batch_size
    #         yield convert_examples_to_features(
    #             self.args, data, self.vocab_src, self.vocab_tgt, with_S=self.with_S, with_E=self.with_E
    #         )
    #     raise StopIteration


def convert_examples_to_features(args, examples, vocab_src, vocab_tgt, with_S=False, with_E=False):
    features = []
    for i, example in enumerate(examples):
        if i % 10000 == 0:
            print("Have convert {} lines to features".format(i))
        src_seq, tgt_seq = example[0], example[1]
        len_src = len(src_seq) + (1 if with_S else 0) + (1 if with_E else 0)
        len_tgt = len(tgt_seq) + (1 if with_S else 0) + (1 if with_E else 0)

        src = ([vocab_src.start_idx] if with_S else []) + [vocab_src.token2idx(w) for w in src_seq] + (
            [vocab_src.end_idx] if with_E else []) + ([vocab_src.padding_idx] * (args.max_length - len_src))

        tgt = ([vocab_tgt.start_idx] if with_S else []) + [vocab_tgt.token2idx(w) for w in tgt_seq] + (
            [vocab_tgt.end_idx] if with_E else []) + ([vocab_tgt.padding_idx] * (args.max_length - len_tgt))

        features.append(InputFeatures(src=src, tgt=tgt))
    all_input_src = torch.tensor([f.src for f in features], dtype=torch.long)
    all_input_tgt = torch.tensor([f.tgt for f in features], dtype=torch.long)
    # dataset = [all_input_src, all_input_tgt]
    dataset = TensorDataset(all_input_src, all_input_tgt)
    return dataset


# class DataLoader(object):
#     def __init__(self, filename, vocab_src=None, vocab_tgt=None, batch_size=128, for_train=True, num_part=5):
#         self.filename = filename
#         self.file = open(self.filename, "r", encoding="utf-8", errors='ignore')
#         self.corpus_lines = self.get_corpus_lines(self.file)
#         self.partition_size = int(self.corpus_lines / num_part)
#         self.batch_size = batch_size
#         self.vocab_src = vocab_src
#         self.vocab_tgt = vocab_tgt
#         self.train = for_train
#
#     def get_corpus_lines(self, file):
#         corpus_lines = 0
#         for index, line in enumerate(file):
#             if line:
#                 corpus_lines = corpus_lines + 1
#         self.file.close()
#         self.file = open(self.filename, "r", encoding="utf-8", errors='ignore')
#         return corpus_lines
#
#     def get_a_example(self, line):
#         d = [x.split() for x in line.strip().split('|')]
#         skip = not (len(d) == 4)
#         for j, i in enumerate(d):
#             if not self.train:
#                 d[j] = i[:30]
#                 if len(d[j]) == 0:
#                     d[j] = [UNK]
#             if len(i) == 0 or len(i) > 30:
#                 skip = True
#         if not (skip and self.train):
#             return d
#         else:
#             return None
#
#     def get_examples(self, idx, partition_size):
#         examples = []
#         end = min((idx + 1) * partition_size, self.corpus_lines)
#         for guid in range(idx * partition_size, end):
#             if guid % self.partition_size == 0:
#                 print("Have processed {} lines to examples".format(guid))
#             try:
#                 line = next(self.file)
#             except StopIteration:
#                 self.file.close()
#                 self.file = open(self.filename, "r", encoding="utf-8", errors='ignore')
#                 line = next(self.file)
#             if line is None:
#                 continue
#             example = self.get_a_example(line)
#             if example:
#                 examples.append(example)
#         return examples
#
#     def get_next_part(self, idx):
#         examples = self.get_examples(idx, self.partition_size)
#         random.shuffle(examples)
#
#         return examples
#
#
#     def __iter__(self):
#         idx = list(range(len(self.data)))
#         if self.train:
#             random.shuffle(idx)
#         cur = 0
#         while cur < len(idx):
#             data = [self.data[i] for i in idx[cur:cur + self.batch_size]]
#             cur += self.batch_size
#             yield batchify(data, self.vocab_src, self.vocab_tgt)
#         raise StopIteration
