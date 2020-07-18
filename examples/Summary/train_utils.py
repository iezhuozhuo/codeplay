# encoding utf-8
import os
import random
import shutil
import logging
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torch.autograd import Variable

from source.utils.engine import Trainer


class trainer(Trainer):
    def __init__(self, args, model, optimizer, train_iter, valid_iter, logger, valid_metric_name="-loss", save_dir=None,
                 num_epochs=5, log_steps=None, valid_steps=None, grad_clip=None, lr_scheduler=None, save_summary=False):

        super().__init__(args, model, optimizer, train_iter, valid_iter, logger, valid_metric_name, num_epochs,
                         save_dir, log_steps, valid_steps, grad_clip, lr_scheduler, save_summary)

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

        tr_loss = 0
        for batch_id, batch in enumerate(self.train_iter, 1):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            article_ids, article_len, article_mask, article_ids_extend_vocab, \
            summary_input_ids, summary_taget_ids, summary_len, summary_mask, extra_zeros = batch
            h_context = Variable(torch.zeros((article_ids.size(0), 2 * self.args.hidden_size))).to(self.args.device)
            coverage = Variable(torch.zeros(article_ids.size())).to(self.args.device)

            loss = self.model(article_ids, article_len, article_mask, article_ids_extend_vocab,
                              summary_input_ids, summary_taget_ids, summary_mask, summary_len,
                              h_context, extra_zeros, coverage)

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

            tr_loss += loss.item()

            if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                self.global_step += 1

            if self.global_step % self.log_steps == 0:
                # logging_loss = tr_loss / self.global_step
                self.logger.info("the current train_steps is {}".format(self.global_step))
                self.logger.info("the current logging_loss is {}".format(loss.item()))

            if self.global_step % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)

                if isinstance(self.model, torch.nn.DataParallel):
                    model = self.model.module
                else:
                    model = self.model
                model.to(self.args.device)

                metrics = evaluate(self.args, model, self.valid_iter, self.logger)

                cur_valid_metric = metrics[self.valid_metric_name]
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                self.logger.info("-" * 85 + "\n")


    def train(self):
        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(self.train_iter) // self.args.gradient_accumulation_steps) + 1
        else:
            self.t_total = len(self.train_iter) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

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
        self.logger.info("Total optimization steps = %d", self.t_total)
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
                self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        for _ in range(int(self.num_epochs)):
            self.train_epoch()

    def save(self, is_best=False, save_mode="best"):
        model_file_name = "state_epoch_{}.model".format(self.epoch) if save_mode == "all" else "state.model"
        model_file = os.path.join(
            self.save_dir, model_file_name)
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file_name = "state_epoch_{}.train".format(self.epoch) if save_mode == "all" else "state.train"
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
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, model_file, train_file):

        model_state_dict = {k.replace('module.', ''): v for k, v in torch.load(model_file).items()}

        # model_state_dict = torch.load(
        #     model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        if self.optimizer is not None and "optimizer" in train_state_dict:
            self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate(args, model, valid_dataset, logger):
    eval_loss, nb_eval_steps = 0.0, 0
    model.eval()
    for batch_id, batch in enumerate(valid_dataset, 1):
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        article_ids, article_len, article_mask, article_ids_extend_vocab, \
        summary_input_ids, summary_taget_ids, summary_len, summary_mask, extra_zeros = batch
        h_context = Variable(torch.zeros((article_ids.size(0), 2 * args.hidden_size))).to(args.device)
        coverage = Variable(torch.zeros(article_ids.size())).to(args.device)

        with torch.no_grad():
            tmp_eval_loss = model(article_ids, article_len, article_mask, article_ids_extend_vocab,
                              summary_input_ids, summary_taget_ids, summary_mask, summary_len,
                              h_context, extra_zeros, coverage)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu.
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    metrics = {"loss": eval_loss}
    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key.upper(), str(metrics[key]))
    return metrics


