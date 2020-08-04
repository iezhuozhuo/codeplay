import os
import random
import shutil
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from tensorboardX import SummaryWriter

from preprocessing_zh import bert_extract_item
from Metric import Metrics, SpanEntityScore, cal_performance

from source.utils.engine import Trainer
import source.utils.Constant as constants


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def checkoutput_and_setcuda(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    return args


def init_logger(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    return logger


def load_pretrain_embed(pretrain_dir, output_dir, word_to_id, emb_dim=300):
    filename_trimmed_dir = os.path.join(output_dir, "embedding_SougouNews.npz")

    if os.path.exists(filename_trimmed_dir):
        embeddings = np.load(filename_trimmed_dir)["embeddings"].astype('float32')
        return embeddings

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    with open(pretrain_dir, "r", encoding='UTF-8') as f:
        for i, line in enumerate(f.readlines()):
            lin = line.strip().split(" ")
            if lin[0] in word_to_id:
                idx = word_to_id[lin[0]]
                emb = [float(x) for x in lin[1:301]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
        np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    return embeddings


def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]

    if args.model_type == "bertsofmax":
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    elif args.model_type == "bertcrf":
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate}
        ]
    else:
        bert_parameters = model.bert.named_parameters()
        start_parameters = model.start_fc.named_parameters()
        end_parameters = model.end_fc.named_parameters()
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': args.learning_rate},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': args.learning_rate},

            {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': 0.001},
            {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001},

            {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay, 'lr': 0.001},
            {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': 0.001},
        ]
    return optimizer_grouped_parameters


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
            if self.args.model_type == "bertsoftmax" or self.args.model_type == "bertcrf":
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.args.model_type != "distilbert":
                    # XLM and RoBERTa don't use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.args.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
            elif self.args.model_type == "bertspan":
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                          "start_positions": batch[3], "end_positions": batch[4]}
                if self.args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.args.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
            else:
                inputs_id, inputs_label, inputs_len, *_ = tuple(t.to(self.args.device) for t in batch)
                outputs = self.model(inputs=(inputs_id, inputs_len) + tuple(_), labels=inputs_label)
                # outputs = self.model(inputs=inputs_id, labels=inputs_label)

            loss = outputs[0]

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
                    model.to(self.args.device)
                else:
                    model = self.model
                    model.to(self.args.device)

                if self.args.model_type == "bertsoftmax" or self.args.model_type == "bertcrf":
                    metrics = evaluate_bert_normal(self.args, model, self.valid_iter, self.logger)
                elif self.args.model_type == "bertspan":
                    metrics = evaluate_bert_span(self.args, model, self.valid_iter, self.logger)
                else:
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
        model_file_name = "state_epoch_{}.model".format(self.epoch) if save_mode == "all" else "state.model"
        model_file = self.save_dir
        model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
        if "bert" in self.args.model_type:
            model_to_save.save_pretrained(model_file)
        else:
            torch.save(model_to_save, model_file)
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
        if "bert" in self.args.model_type:
            self.model = self.model.from_pretrained(model_file)
        else:
            model_state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
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
    labels, preds = [], []
    model.eval()
    for batch in valid_dataset:
        inputs_id, inputs_label, inputs_len = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            tmp_eval_loss, logits = model(inputs=(inputs_id, inputs_len), labels=inputs_label)

            if getattr(args, "optimized"):
                pred = model.crf.decode(logits)
                tags = pred.squeeze(0).cpu().numpy().tolist()
            else:
                tags, _ = model.crf._obtain_labels(logits, args.id2label, inputs_len)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        out_label_ids = inputs_label.cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1, temp_2 = [], []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif out_label_ids[i][j] == args.label2id[constants.SEP]:
                    labels.append(temp_1)
                    preds.append(temp_2)
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    if args.optimized:
                        temp_2.append(args.id2label[tags[i][j]])
                    else:
                        temp_2.append(tags[i][j])

    # # 其他评估
    # metrics = Metrics(labels, preds)
    # metrics.report_scores()
    # # metrics.report_confusion_matrix()
    # results = {"f1": metrics.avg_metrics['f1_score'], "acc": metrics.precision_scores}
    # results.update({"loss": eval_loss})
    # return results

    # seqeval评估
    metrics = cal_performance(preds, labels)
    metrics.update({"loss": eval_loss})
    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key.upper(), str(metrics[key]))
    return metrics


def evaluate_bert_normal(args, model, valid_dataset, logger):
    eval_loss = 0.0
    nb_eval_steps = 0
    labels, preds = [], []
    for step, batch in enumerate(valid_dataset):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if "crf" in args.model_type:
            logger.info("Using CRF Evaluation")
            tags = model.crf.decode(logits, inputs['attention_mask'])
            pred = tags.squeeze(0).cpu().numpy().tolist()
        else:
            pred = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1, temp_2 = [], []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif out_label_ids[i][j] == args.label2id[constants.SEP]:
                    labels.append(temp_1)
                    preds.append(temp_2)
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[pred[i][j]])

    ## 其他评估
    # metrics = Metrics(labels, preds)
    # metrics.report_scores()
    # # metrics.report_confusion_matrix()
    # results = {"f1": metrics.avg_metrics['f1_score'], "acc": metrics.precision_scores}
    # results.update({"loss": eval_loss})
    # return results

    # seqeval评估
    metrics = cal_performance(preds, labels)
    metrics.update({"loss": eval_loss})
    for key in sorted(metrics.keys()):
        logger.info("  %s = %s", key.upper(), str(metrics[key]))
    return metrics


# TODO 评估需要处理
def evaluate_bert_span(args, model, eval_features, logger):

    metric = SpanEntityScore(args.id2label)
    # Eval!
    eval_loss, nb_eval_steps = 0.0, 0
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        subjects = f.subjects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids, "end_positions": end_ids}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        R = bert_extract_item(start_logits, end_logits)
        T = subjects
        metric.update(true_subject=T, pred_subject=R)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss

    for key in sorted(results.keys()):
        logger.info("  %s = %s", key.upper(), str(results[key]))
    return results