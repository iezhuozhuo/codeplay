# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!
import os
import re
import json
import jieba
import random
import codecs
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from source.utils.misc import init_logger, timer
import source.utils.Constant as constants
from source.inputters.field import TextField, NumberField

logger = init_logger()

user_dict_name = "/home/gong/NLPData/JDDC/all_dict.txt"
logger.info("loading {} user_dict".format(user_dict_name))
jieba.load_userdict(user_dict_name)


# 定义输入的Example类
class Example(object):
    def __init__(self, question, answer, q_len, a_len):
        self.question = question
        self.answer = answer
        self.q_len = q_len
        self.a_len = a_len


# 定义输入feature类
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_id,
                 input_len,
                 input_mask,
                 label):
        self.input_id = input_id
        self.input_len = input_len
        self.input_mask = input_mask
        self.label = label


# 定义任务的预料处理器 Processor
class MTDiaCorpus(object):
    def __init__(self,
                 args,
                 max_vocab_size=50000,
                 min_freq=1,
                 specials=None,
                 share_vocab=True):
        super(MTDiaCorpus, self).__init__()

        self.args = args
        self.data_file = os.path.join(args.data_dir, "data.pt")
        self.field_question_file = os.path.join(args.output_dir, "field_question.pt")
        self.field_answer_file = os.path.join(args.output_dir, "field_answer.pt")
        self.max_vocab_size = max_vocab_size

        self.min_freq = min_freq
        self.specials = specials
        # self.tokenizer = self.get_tokenizer()

        logger.info("Initial Corpus ...")
        self.field = {"question": TextField(tokenize_fn=None, special_tokens=self.specials),
                      "answer": TextField(tokenize_fn=None, special_tokens=self.specials)}

        if share_vocab:
            self.field["answer"] = self.field["question"]

        self.load()

    def load(self):
        if not os.path.exists(self.data_file):
            logger.info("Build Corpus ...")
            self.build()
        else:
            self.load_data(self.data_file)
            self.load_field()

    def build(self):
        data_question_file = os.path.join(self.args.data_dir, self.args.question_file)
        data_answer_file = os.path.join(self.args.data_dir, self.args.answer_file)

        # check data and whether to pre-process data
        self.check(data_question_file, data_answer_file)

        logger.info("Reading Data ...")
        data_raw = self.read_data(data_question_file, data_answer_file, data_type="train")
        random.shuffle(data_raw)
        train_raw = data_raw[:int(len(data_raw) * 0.8)]
        valid_raw = data_raw[int(len(data_raw) * 0.8):int(len(data_raw) * 0.9)]
        test_raw = data_raw[int(len(data_raw) * 0.9):]

        # 根据训练集来定制词表
        logger.info("Build Vocab from {} and {} ...".format(data_question_file, data_answer_file))
        self.build_vocab(train_raw)

        train_data = self.build_examples(train_raw, data_type="train")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        test_data = self.build_examples(test_raw, data_type="test")

        self.data = {"train": train_data,
                     "valid": valid_data,
                     "test": test_data
                     }

        logger.info("Saved text field to '{}'".format(self.field_question_file))
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        field_text = {"itos": self.field["question"].itos,
                      "stoi": self.field["question"].stoi,
                      "vocab_size": self.field["question"].vocab_size,
                      "specials": self.field["question"].specials
                      }
        torch.save(field_text, self.field_question_file)

        logger.info("Saved text field to '{}'".format(self.field_answer_file))
        field_text = {"itos": self.field["answer"].itos,
                      "stoi": self.field["answer"].stoi,
                      "vocab_size": self.field["answer"].vocab_size,
                      "specials": self.field["answer"].specials
                      }
        torch.save(field_text, self.field_answer_file)

        logger.info("Saved data to '{}'".format(self.data_file))
        torch.save(self.data, self.data_file)

    def check(self, data_question_file, data_answer_file):

        if not os.path.isfile(data_question_file) or not os.path.isfile(data_answer_file):
            logger.info("pre-process data question and answer can't find and preprocess the origin data")
            origin_data_file = os.path.join(self.args.data_dir, self.args.origin_file)
            if not os.path.isfile(origin_data_file):
                logger.info("Origin data doesn't find")
                return
            self.preprocess(origin_data_file, data_question_file, data_answer_file)
        else:
            logger.info("question and answer files can find")

    def preprocess(self, origin_data_file, data_question_file, data_answer_file):
        sessionId = self.args.first_sessionId  # "00029c51f92e8f34250d6af329c9a8df"
        question, answer, QAQAQ = '', '', ''
        countQuestion, countAnswer = 0, 0
        with codecs.open(origin_data_file, mode="r", encoding="utf-8") as rf:
            try:
                line = rf.readline()
                while line:
                    splitline = line.strip('\r\n').split("\t")
                    if sessionId == splitline[0]:
                        with codecs.open(data_question_file, mode="a",
                                         encoding="utf-8") as wf_question:
                            with codecs.open(data_answer_file, mode="a",
                                             encoding="utf-8") as wf_answer:
                                try:
                                    if splitline[2] == '0':
                                        if countQuestion == 3 and countAnswer == 2:
                                            wf_question.write(QAQAQ + "\n")
                                            wf_answer.write(answer + "\n")
                                            question = ''
                                            answer = ''
                                            QAQAQ = ''
                                            countQuestion = 0
                                            countAnswer = 0

                                        if answer != '':
                                            # answer = answer.strip(',')
                                            # wf_question.write(answer)
                                            QAQAQ = QAQAQ + answer
                                            answer = ''
                                            countAnswer = countAnswer + 1
                                        question = question + splitline[6] + ' _EOS_ '

                                    elif splitline[2] == '1':
                                        if question != '':
                                            # question = question.strip(',')
                                            # wf_question.write(question)
                                            QAQAQ = QAQAQ + question
                                            question = ''
                                            countQuestion = countQuestion + 1
                                        answer = answer + splitline[6] + ' _EOS_ '

                                except Exception as e:
                                    logger.error("data_processing:write into chatmasked_user failure", e)
                                finally:
                                    wf_question.close()
                                    wf_answer.close()

                    else:
                        sessionId = splitline[0]
                        question, answer, QAQAQ = '', '', ''
                        countQuestion, countAnswer = 0, 0
                        continue

                    line = rf.readline()

            except Exception as e:
                logger.error("data_processing: data processing failure!", e)
            finally:
                rf.close()

    @timer
    def read_data(self, data_question_file, data_answer_file, data_type="train"):
        """
        读取样本文件
        Return:
            data: 字典列表，每个字典由 question, answer
        """
        if not os.path.isfile(data_question_file) or not os.path.isfile(data_answer_file):
            logger.info("{} data question and answer can't find".format(data_type))
            return None

        f_question = open(data_question_file, 'r', encoding="utf-8")
        f_answer = open(data_answer_file, 'r', encoding="utf-8")

        lines = []
        for i, (question, answer) in enumerate(zip(f_question, f_answer)):
            # FIXME 全量数据
            if i % 10000 == 0:
                logger.info("Read {} examples from {}".format(len(lines), data_type.upper()))
            # if len(lines) >= 20000:
            #     break
            question_tokens = self.tokenizer(question)
            answer_tokens = self.tokenizer(answer)
            lines.append({"answer": " ".join(answer_tokens), "question": " ".join(question_tokens)})

        logger.info("Read total {} examples from {}".format(len(lines), data_type.upper()))
        f_question.close()
        f_answer.close()
        return lines

    def build_examples(self, data_raw, data_type="train"):
        if data_raw == None:
            logger.info("{} data text and label can't find".format(data_type))

        examples, len_seq_question, len_seq_answer = [], [], []
        desc_message = "GET DATA FROM " + data_type.upper()
        for data in tqdm(data_raw, desc=desc_message):
            len_seq_question.append(len(str.split(data["question"])))
            len_seq_answer.append(len(str.split(data["answer"])))

            examples.append(Example(
                question=data["question"],
                answer=data["answer"],
                q_len=len_seq_question[-1],
                a_len=len_seq_answer[-1])
            )

        len_seq_left = np.array(len_seq_question)
        len_seq_right = np.array(len_seq_answer)
        logger.info("left {} sequence length converge 95%".format(np.percentile(len_seq_left, 95)))
        logger.info("right {} sequence length converge 95%".format(np.percentile(len_seq_right, 95)))

        return examples

    def build_vocab(self, data):
        """
        从train的text分别生成字典
        data format [{"question":, "answer":},...]
        """

        xs = [[x["question"], x["answer"]] for x in data]
        self.field["question"].build_vocab(xs,
                                           min_freq=self.min_freq,
                                           max_size=self.max_vocab_size)

    def load_data(self, data_file=None):
        """ 加载.pt格式的语料 """
        prepared_data_file = data_file or self.data_file
        logger.info("Loading prepared data from {} ...".format(prepared_data_file))
        self.data = torch.load(prepared_data_file)
        # logger.info("Number of examples:",
        #       " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_field(self):
        question_field = torch.load(self.field_question_file)
        answer_field = torch.load(self.field_answer_file)
        self.field["question"].load(question_field)
        self.field["answer"].load(answer_field)

    def create_batch(self, data_type="train"):
        examples = self.data[data_type]
        # FIXME Check example num
        # examples = examples[0:1024]
        features_cache_path = os.path.join(
            self.args.data_dir,
            "features-{}-{}-{}.pt".format(data_type, self.args.max_seq_length, "aug" if self.args.aug else "no-aug")
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading prepared features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert examples to features")
            features = self.convert_examples_to_features(examples)
            torch.save(features, features_cache_path)
        dataset = None  # 记得delete
        # 按需修改
        # all_left_id = torch.tensor([f.left_ids for f in features], dtype=torch.long)
        # all_right_id = torch.tensor([f.right_ids for f in features], dtype=torch.long)
        # all_left_char_id = torch.tensor([f.left_char_ids for f in features], dtype=torch.long)
        # all_right_char_id = torch.tensor([f.right_char_ids for f in features], dtype=torch.long)
        # all_left_len = torch.tensor([f.left_len for f in features], dtype=torch.long)
        # all_right_len = torch.tensor([f.right_len for f in features], dtype=torch.long)
        # all_left_char_len = torch.tensor([f.left_chars_len for f in features], dtype=torch.long)
        # all_right_char_len = torch.tensor([f.right_chars_len for f in features], dtype=torch.long)
        # all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_left_id, all_right_id, all_left_len, all_right_len, all_left_char_id, all_right_char_id, all_left_char_len, all_right_char_len, all_label)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        return dataloader

    def convert_examples_to_features(self, examples, data_type="train"):
        features = []
        text_q_len, text_a_len = [], []
        desc_message = "GET DATA FROM " + data_type.upper()
        for example in tqdm(examples, desc=desc_message):
            question_words = str.split(example.question)
            answer_words = str.split(example.answer)


            # process the encoder inputs
            if len(question_words) > self.args.max_enc_seq_length:
                question_words = question_words[:self.args.max_enc_seq_length]
            question_words_len = len(question_words)

            if len(answer_words) > self.args.max_dec_seq_length:
                answer_words = answer_words[:self.args.max_dec_seq_length]
            answer_words_len = len(answer_words)

            question_ids, answer_ids = [], []
            for i, word in enumerate(question_words):
                question_ids.append(
                    self.field["qustion"].stoi.get(word, self.field["question"].stoi.get(constants.UNK_WORD)))
            for i, word in enumerate(answer_words):
                answer_ids.append(
                    self.field["answer"].stoi.get(word, self.field["answer"].stoi.get(constants.UNK_WORD)))

            # summary_input_ids = [self.field["summary"].stoi[constants.BOS_WORD]] + summary_ids
            # summary_taget_ids = summary_ids[:]
            # if len(summary_input_ids) > self.args.max_dec_seq_length:
            #     summary_input_ids = summary_input_ids[: self.args.max_dec_seq_length]  # 无结束标志
            #     summary_taget_ids = summary_taget_ids[: self.args.max_dec_seq_length]
            # else:
            #     summary_taget_ids.append(self.field["summary"].stoi[constants.EOS_WORD])  # 无截断有结束标志
            # assert len(summary_input_ids) == len(summary_taget_ids)
            # summary_len = len(summary_input_ids)

        return features

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

    def replace_special_token(self, sentence):
        """
        特殊字段有：
        1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
        2. [ORDERID_10187709] —— 订单号
        3. [数字x] —— 数字
        4. https://item.jd.com/5898522.html —— 网址
        5. [地址x] —— 地址
        6. [链接x] —— 链接
        7. [金额x] —— 金额
        8. [日期x] —— 日期
        9. [时间x] —— 时间
        10. [站点x] —— 站点
        11. [组织机构x] ——组织机构
        12. [电话x] —— 电话
        13. [姓名x] —— 人名
        对于表情，做法是直接删除。其他用希腊符号替换。
        """
        sentence = re.sub(" _EOS_ ", " ", sentence)
        sentence = re.sub(
            "#E\-[\w]*(抱拳|傲慢|得意|蛋糕|呕吐|闭嘴|礼物|yaoping|柠檬|流泪|怒火|撇嘴|太阳|咒骂|糗|猪猪|足球|磕头|大兵|电话|灯泡|飞鸟|奋斗|高兴|击打|饥饿|咖啡|口罩|骷髅|可乐|疯狂|白眼|阴险|叹气|奸笑|发呆|害羞|飞吻|怒火|悲伤|胜利|生病|弱|可怜|咖啡|酷酷|眩晕|流泪|发抖|难过|右哼哼|惊恐|悲伤|犯困|愤怒|凋谢|哈欠|拥抱|抓狂|鄙视|时间|啤酒|勾引|左哼哼|月亮|偷笑|震惊|惊讶|跳跳|瞌睡|可爱|衰样|好|憨笑|水果|色色|黑线|微笑|流汗|握手|心碎|问号|大哭|亲亲|抠鼻|拜拜|鬼脸|香吻|米饭|花朵|尴尬|擦汗|安慰|委屈|调皮|爱心|我一定尽力为您解答的哦|很棒|鼓掌)+",
            "表情", sentence)  ## 匹配 #E-流汗
        sentence = re.sub("#E\-[\w]+\[数字x]", "表情", sentence)
        # sentence = re.sub("\[ORDERID_[\d]+]", "[订单x]", sentence)
        sentence = re.sub("\[ORDERID_[\d]+]", "订单", sentence)
        sentence = re.sub("\[数字x]", "数字", sentence)
        sentence = re.sub("\[地址x]", "地址", sentence)
        sentence = re.sub("\[链接x]", "链接", sentence)
        sentence = re.sub("\[金额x]", "金额", sentence)
        sentence = re.sub("\[日期x]", "日期", sentence)
        sentence = re.sub("\[时间x]", "时间", sentence)
        sentence = re.sub("\[站点x]", "站点", sentence)
        sentence = re.sub("\[组织机构x]", "组织机构", sentence)
        sentence = re.sub("\[电话x]", "电话", sentence)
        sentence = re.sub("\[姓名x]", "姓名", sentence)
        sentence = re.sub("\[邮箱x]", "邮箱", sentence)
        sentence = re.sub("\[身份证号x]", "身份证号", sentence)
        sentence = re.sub("\[商品快照]", "商品快照", sentence)
        sentence = re.sub("\[表情]", "表情", sentence)
        sentence = re.sub(
            "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", "链接",
            sentence)
        sentence = re.sub("(http|ftp|https):\/\/ε", "链接", sentence)
        sentence = re.sub("[\d]+.*[\d]+", "数字", sentence)
        sentence = re.sub("【收到不支持的消息类型，暂无法显示】", " ", sentence)

        # sentence = re.sub("#E\-[s]*(ν|γ|π|ζ|ρ|α|ε)*", "α", sentence)
        # sentence = re.sub("α", " ", sentence)
        # sentence = re.sub("ε", "[链接x]", sentence)
        # sentence = re.sub("γ", "[数字x]", sentence)

        return sentence

    def tokenizer(self, line):
        line = self.replace_special_token(line)
        words = jieba.cut(line.strip())
        word_list = list(words)
        # jieba.disable_parallel()
        return word_list


# 定义gpt2数据读入格式
class GPT2Example(object):
    def __init__(self, sessionId, session):
        self.sessionId = sessionId
        self.session = session


class GPT2Feature(object):
    def __init__(self, session_ids):
        self.session_ids = session_ids


class GPTProcessor(object):
    def __init__(self):
        super(GPTProcessor, self).__init__()

    def get_train_and_dev_examples(self, origin_data_path):
        origin_data_file = os.path.join(origin_data_path, "jdcc_sessions_small.pkl")
        logger.info("Create train and dex examples from {}".format(origin_data_path))
        train_examples, dex_examples = self._create_examples(self._read_pkl(origin_data_file))
        return train_examples, dex_examples

    def _read_pkl(self, input_file):
        with open(input_file, "rb") as f:
            data = pkl.load(f)
        return data

    def _create_examples(self, sessions):
        examples = []
        for sessionId, session in enumerate(sessions):
            examples.append(GPT2Example(sessionId=sessionId, session=session))
        train_examples, dex_examples = train_test_split(examples, test_size=0.001, random_state=42)
        return train_examples, dex_examples


def convert_examples_to_features_gpt(examples, tokenizer, max_seq_length,
                                     set_type="train", split_token=" _EOS_ "):
    features = []
    logger.info("Convert {} examples to features".format(set_type))
    for example in tqdm(examples):
        utterances = [utterance for utterance in example.session.split(split_token) if utterance != ""]
        session_ids = [tokenizer.cls_token_id]
        for utterance in utterances:
            session_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
            session_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
        # 对超过max_seq_length的长度进行截断
        session_ids = session_ids[:max_seq_length]
        features.append(GPT2Feature(session_ids=session_ids))
    return features


class GPT2Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        # input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def collate_fn_gpt(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    pad_id = 0
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def load_and_cache_examples_gpt(args, processor, tokenizer, logger, data_type='train'):
    cached_features_file = os.path.join(args.data_dir, 'cached-{}-{}-small'.format(
                                                        args.model_type,
                                                        str(args.max_seq_length),))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)  # different from bert process
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        train_examples, dex_examples = processor.get_train_and_dev_examples(args.data_dir)
        train_features = convert_examples_to_features_gpt(train_examples, tokenizer, args.max_seq_length)
        dev_features = convert_examples_to_features_gpt(dex_examples, tokenizer, args.max_seq_length)
        features = {"train": train_features, "dev": dev_features}
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    target_features = features[data_type]
    datas = [f.session_ids for f in target_features]
    target_dataset = GPT2Dataset(datas)
    return target_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/gong/NLPData/JDDC",
        type=str,
    )
    parser.add_argument(
        "--origin_file",
        default="chat.txt",
        type=str,
    )
    parser.add_argument(
        "--first_sessionId",
        default="00029c51f92e8f34250d6af329c9a8df",
        type=str,
    )

    parser.add_argument(
        "--question_file",
        default="question.txt",
        type=str,
    )
    parser.add_argument(
        "--answer_file",
        default="answer.txt",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="/home/gong/zz/data/jddc/",
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--max_char_seq_length",
        default=5,
        type=int,
    )
    parser.add_argument("--aug", action="store_true")

    args, _ = parser.parse_known_args()
    # print(args)
    processor = MTDiaCorpus(args)
