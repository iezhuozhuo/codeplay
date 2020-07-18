# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

import re
import random

import torch


class Example(object):
    def __init__(self, text_a, text_b, label):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def read_examples(data_path, data_type="train"):
    data_examples = torch.load(data_path)
    examples = data_examples[data_type]
    examples_pos, examples_neg = [], []
    for example in examples:
        if example.label == '0':
            examples_neg.append(example)
        else:
            examples_pos.append(example)
    print("Pos num: {}, Neg num: {}, pos/neg = {}".format(len(examples_pos), len(examples_neg),
                                                          len(examples_pos) / len(examples_neg)))
    return examples_pos, examples_neg, data_examples


# 交换词
def random_swap(words_left, words_right, left_n, right_n):
    new_words_left = words_left.copy()
    for _ in range(left_n):
        new_words_left = swap_word(new_words_left)

    new_words_right = words_right.copy()
    for _ in range(right_n):
        new_words_right = swap_word(new_words_right)

    return new_words_left, new_words_right


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


# 随机删除词
def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def edu(example, alpha_rs=0.1, num_aug=4):
    left = example.text_a
    right = example.text_b
    words_left = left.split()
    words_right = right.split()

    num_words_left = len(words_left)
    num_words_right = len(words_right)

    left_n_rs = max(1, int(alpha_rs * num_words_left))
    right_n_rs = max(1, int(alpha_rs * num_words_right))

    augmented_examples = []
    while len(augmented_examples) < num_aug:
        p = random.uniform(0, 1)
        if p <= 0.1:
            words_left_delete = random_deletion(words_left, 0.2)
            words_right_delete = random_deletion(words_right, 0.2)
            augmented_examples.append(Example(
                text_a=" ".join(words_left_delete),
                text_b=" ".join(words_right_delete),
                label=example.label
            ))
        elif 0.1 < p <= 0.5:
            new_words_left, new_words_right = random_swap(words_left, words_right, left_n_rs, right_n_rs)
            augmented_examples.append(Example(
                text_a=" ".join(new_words_left),
                text_b=" ".join(new_words_right),
                label=example.label
            ))
        else:
            augmented_examples.append(example)

    return augmented_examples


def augment(data_path, data_aug_path, data_type="train"):
    examples_pos, examples_neg, data_examples = read_examples(data_path, data_type)
    num = int((len(examples_neg) - len(examples_pos)) / len(examples_pos)) + 2
    new_examples = []
    for example in examples_pos:
        augmented_examples = edu(example, num_aug=num)
        new_examples += augmented_examples
    random.shuffle(new_examples)
    print("augment pos {}".format(len(new_examples)))
    if len(new_examples) > len(examples_neg):
        new_examples = new_examples[0:len(examples_neg)]
    new_examples += examples_neg
    random.shuffle(new_examples)
    data_examples[data_type] = new_examples
    print("train: {}".format(len(new_examples)))
    torch.save(data_examples, data_aug_path)


if __name__ == "__main__":
    data_path = "/home/gong/zz/data/Match/data.pt"
    data_aug_path = "/home/gong/zz/data/Match/data_aug.pt"
    augment(data_path, data_aug_path, "train")



