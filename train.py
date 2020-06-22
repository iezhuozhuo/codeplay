'''
This script handles the training process.
'''

import argparse
import math
import time
import shutil
import dill as pickle
from tqdm import tqdm
import logging
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import source.utils.Constant as Constants
from source.models.transformer import Transformer
from source.modules.optim import ScheduledOptim
from source.utils.engine import Trainer
from source.utils.engine import MetricsManager
from source.utils.misc import Pack


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


class Metrics(MetricsManager):
    def __init__(self):
        super(Metrics, self).__init__()
        self.total_loss = 0
        self.n_word_total = 0
        self.n_word_correct = 0

    def update(self, metric):
        self.n_word_total += metric.n_word
        self.n_word_correct += metric.n_correct
        self.total_loss += metric.loss.item()

        loss_per_word = self.total_loss / self.n_word_total
        accuracy = self.n_word_correct / self.n_word_total
        self.metrics_cum["loss_per_word"] = loss_per_word
        self.metrics_cum["accuracy"] = accuracy
        self.metrics_cum["ppl"] = math.exp(min(loss_per_word, 100))

    def get(self, name):
        val = self.metrics_cum[name]
        return val

    def report_cum(self):
        metric_strs = []
        for key, val in self.metrics_cum.items():
            metric_str = "{}-{:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def collect_metrics(outputs, target, trg_pad_idx, smoothing):
    metrics = Pack()
    loss, n_correct, n_word = cal_performance(
        outputs, target, trg_pad_idx, smoothing=smoothing)

    metrics.add(n_correct=n_correct)
    metrics.add(n_word=n_word)
    metrics.add(loss=loss)
    return metrics


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


class trainer(Trainer):
    def __init__(self,
                 args,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 # generator=None,
                 valid_metric_name="-loss_per_word",
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 lr_scheduler=None,
                 save_summary=False):
        super(Trainer).__init__()

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        # self.generator = generator
        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = self.args.epoch
        self.save_dir = save_dir if save_dir else self.args.save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        # self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary

        if self.save_summary:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0
        self.init_message()

    def train_epoch(self):
        self.epoch += 1
        train_start_message = "Training Epoch - {}".format(self.epoch)
        self.logger.info(train_start_message)

        train_mm = Metrics()
        num_batches = len(self.train_iter)
        if self.log_steps is None:
            # 要考虑多GPU情况
            self.log_steps = num_batches % 5
        if self.valid_steps is None:
            self.valid_steps = num_batches

        # train_iterator = tqdm(self.train_iter, mininterval=2, desc=train_start_message, leave=False)
        for batch_id, batch in enumerate(self.train_iter, 1):

            self.model.train()
            # prepare data
            src_seq = patch_src(batch.src, self.model.src_pad_idx).to(self.args.device)
            trg_seq, gold = map(lambda x: x.to(self.args.device), patch_trg(batch.trg, self.model.trg_pad_idx))

            self.optimizer.zero_grad()
            pred = self.model(src_seq, trg_seq)
            metrics = collect_metrics(pred, gold, self.model.trg_pad_idx, True)
            loss = metrics.loss
            loss.backward()
            self.optimizer.step_and_update_lr()
            train_mm.update(metrics)

            self.batch_num += 1
            if batch_id % self.log_steps == 0:
                message_prefix = "[Train][{}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_cum()
                self.logger.info("   ".join(
                    [message_prefix, metrics_message]))
                if self.save_summary:
                    self.summarize_train_metrics(metrics, self.batch_num)

            if self.valid_steps > 0 and batch_id % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm = evaluate(self.model, self.valid_iter, self.args)
                message_prefix = "[Valid][{}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message]))
                if self.save_summary:
                    self.summarize_valid_metrics(valid_mm, self.batch_num)

                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                self.logger.info("-" * 85 + "\n")

    def train(self):
        for _ in range(self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        model_file_name = "state_epoch_{}.model".format(self.epoch) if self.args.save_mode == "all" else "state.model"
        model_file = os.path.join(
            self.save_dir, model_file_name)
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("[Info] Saved model state to '{}'".format(model_file))

        train_file_name = "state_epoch_{}.train".format(self.epoch) if self.args.save_mode == "all" else "state.train"
        train_file = os.path.join(
            self.save_dir, train_file_name)
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "settings": self.args}
        torch.save(train_state, train_file)
        self.logger.info("[Info] Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "[Info] Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))


def evaluate(model, validation_data, args):
    ''' Epoch operation in evaluation phase '''
    device = args.device
    model.eval()
    valid_mm = Metrics()
    with torch.no_grad():
        for batch in validation_data:
            # prepare data
            src_seq = patch_src(batch.src, model.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, model.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            metrics = collect_metrics(pred, gold, model.trg_pad_idx, False)
            valid_mm.update(metrics)

    return valid_mm


def main():
    '''
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_dir', default=None)
    parser.add_argument('-save_mode', type=str, default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # Logger definition
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    if not opt.log and not opt.save_dir:
        logger.info('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        logger.info('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    logger.info("========= Loading Dataset =========")
    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, opt.device)
    elif opt.data_pkl:
        training_data, validation_data = prepare_dataloaders(opt, opt.device)
    else:
        raise

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(opt.device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, opt.n_warmup_steps)

    logger.info("Training starts ..."+"\n")
    trainer_op = trainer(args=opt,
                        model=transformer,
                        optimizer=optimizer,
                        train_iter=training_data,
                        valid_iter=validation_data,
                        logger=logger)
    trainer_op.train()
    logger.info("Training done!")





if __name__ == '__main__':
    main()
