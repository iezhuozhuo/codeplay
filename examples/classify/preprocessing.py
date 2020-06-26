import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.data import Field, TabularDataset

import source.utils.Constant as constants


class DataFildProcessor():
    def __init__(self, args, ):
        super(DataFildProcessor, self).__init__()
        # self.train_data_dir = os.path.join(args.data_dir, args.train_file)

    def get_train_examples(self, data_dir, train_file):
        pass

    def get_dev_examples(self, data_dir, dev_file):
        pass

    def get_test_examples(self, data_dir, test_file):
        pass


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


if __name__ == "__main__":
    pass