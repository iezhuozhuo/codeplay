# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

import os
import random
import shutil
import logging
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

from source.utils.engine import Trainer


def calculate_loss_and_accuracy(outputs, labels, device, pad_id=0):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


class trainer(Trainer):
    def __init__(self, args, model, optimizer, train_iter, valid_iter, logger, valid_metric_name="-loss", save_dir=None,
                 num_epochs=5, log_steps=None, valid_steps=None, grad_clip=None, lr_scheduler=None, model_log=None, save_summary=False):

        super().__init__(args, model, optimizer, train_iter, valid_iter, logger, valid_metric_name, save_dir,
                         num_epochs, log_steps, valid_steps, grad_clip, lr_scheduler, model_log, save_summary)

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir if save_dir else self.args.output_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary
        self.model_log = model_log

        if self.save_summary:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.global_step = 0
        self.init_message()

    def train_epoch(self):
        self.epoch += 1
        train_start_message = "Training Epoch - {}".format(self.epoch)
        self.logger.info(train_start_message)

        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
        oom_time = 0  # 记录 out of memory的次数
        for batch_id, input_ids in enumerate(self.train_iter):
            self.model.train()
            # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids = input_ids.to(self.args.device)
            # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
            try:
                outputs = self.model.forward(input_ids=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=self.args.device, pad_id=0)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
                else:
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # 进行一定step的梯度累计之后，更新参数
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    tr_loss += loss.item()
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.global_step += 1
                    nb_tr_steps += 1

                    if self.global_step % self.log_steps == 0:
                        # logging_loss = tr_loss / self.global_step
                        self.logger.info("the current train_steps is {}".format(self.global_step))
                        self.logger.info("the current logging_loss is {}".format(loss.item()))

                    if self.global_step % self.valid_steps == 0:
                        self.logger.info(self.valid_start_message)
                        self.model.to(self.args.device)
                        metrics = evaluate(self.args, self.model, self.valid_iter, self.logger)
                        cur_valid_metric = metrics[self.valid_metric_name]
                        if self.is_decreased_valid_metric:
                            is_best = cur_valid_metric < self.best_valid_metric
                        else:
                            is_best = cur_valid_metric > self.best_valid_metric
                        self.save(is_best)
                        self.logger.info("-" * 85 + "\n")

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    self.logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    self.logger.info(str(exception))
                    raise exception

        # loss_epoch = tr_loss / nb_tr_steps
        # self.model_log.add_metric(metric_name='train_loss', metric_value=loss_epoch, epoch=self.epoch)
        # self.model_log.add_metric(
        #     metric_name="lr", metric_value=self.optimizer.state_dict()['param_groups'][0]['lr'], epoch=self.epoch)

    def train(self):
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(self.train_iter) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.train_iter) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        self.logger.info(self.train_start_message)
        self.logger.info("Num examples = %d", len(self.train_iter))
        self.logger.info("Num Epochs = %d", self.num_epochs)
        self.logger.info("Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        self.logger.info(
            "Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        self.logger.info("Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("Total optimization steps = %d", t_total)
        self.logger.info("logger steps = %d", self.log_steps)
        self.logger.info("valid steps = %d", self.valid_steps)
        self.logger.info("-" * 85 + "\n")
        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )

        for _ in range(int(self.num_epochs)):
            self.train_epoch()

    def save(self, is_best=False, save_mode="best"):
        model_file_name = "{}_pytorch_model_epoch_{}.bin".format(self.args.model_type, self.epoch) \
            if save_mode == "all" else "{}_pytorch_model.bin".format(self.args.model_type)
        model_file = os.path.join(
            self.save_dir, model_file_name)
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file_name = "{}_epoch_{}_config".format(self.args.model_type, self.epoch) \
            if save_mode == "all" else "{}_config".format(self.args.model_type)
        train_file = os.path.join(
            self.save_dir, train_file_name)
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict(),
                       "settings": self.args}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best_model.bin")
            best_train_file = os.path.join(self.save_dir, "best_model_config")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, model_file, train_file):
        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate(args, model, valid_dataset, logger):
    eval_loss, eval_acc, nb_eval_steps = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(valid_dataset):
            input_ids = input_ids.to(args.device)
            outputs = model.forward(input_ids=input_ids)
            tmp_eval_loss, temp_eval_acc = calculate_loss_and_accuracy(outputs, labels=input_ids, device=args.device)
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
                temp_eval_acc = temp_eval_acc.mean()
            if args.gradient_accumulation_steps > 1:
                tmp_eval_loss = tmp_eval_loss / args.gradient_accumulation_steps
                temp_eval_acc = temp_eval_acc / args.gradient_accumulation_steps
            eval_loss += tmp_eval_loss.item()
            eval_acc += temp_eval_acc.item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_acc = eval_acc / nb_eval_steps
    metrics = {}
    metrics.update({"loss": eval_loss})
    metrics.update({"acc": eval_acc})
    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key.upper(), str(metrics[key]))
    return metrics