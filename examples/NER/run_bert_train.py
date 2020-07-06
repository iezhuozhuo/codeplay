import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from preprocessing import (CnerProcessor, CnerProcessorSpan, CNerTokenizer, load_and_cache_examples,
                           collate_fn_normal, collate_fn_span)
from train_utils import (
    set_seed, checkoutput_and_setcuda, init_logger, get_optimizer_grouped_parameters,
    trainer, evaluate, evaluate_bert_normal, evaluate_bert_span)
from ModelConfig import BertModelConfig

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.models.ner import BertSoftmaxForNer, BertCrfForNer, BertSpanForNer
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup


MODEL_CLASSES = {
    'bertsoftmax': (BertModelConfig, BertSoftmaxForNer, CNerTokenizer),
    'bertcrf': (BertModelConfig, BertCrfForNer, CNerTokenizer), # bertcrf使用的是crf v2
    'bertspan': (BertModelConfig, BertSpanForNer, CNerTokenizer)
}


ProcessorClass = {
    "bertsoftmax": CnerProcessor,
    'bertcrf': CnerProcessor,
    "bertspan": CnerProcessorSpan,
}


def main():
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    configs, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    args, config_class = configs(parser)

    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)

    # Set seed
    set_seed(args)

    # specials = [constants.UNK_WORD, constants.PAD_WORD, constants.CLS, constants.SEP]
    processor = ProcessorClass[args.model_type]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # 目前版本不能传进去的config未使用的任意参数  unused_kwargs=True可以返回未使用的config
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, loss_type=args.loss_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_hidden_layers=3)
    config.loss_type = args.loss_type
    config.soft_label = True

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None
                                                )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None,)

    logger.info("Training/evaluation parameters %s", args)
    collate_fn = collate_fn_normal if "span" not in args.model_type else collate_fn_span
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, tokenizer, logger, data_type='train')
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collate_fn)

        eval_dataset = load_and_cache_examples(args, processor, tokenizer, logger, data_type='dev')
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if "span" not in args.model_type:
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         collate_fn=collate_fn)
        else:
            eval_dataloader = eval_dataset

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5 if len(train_dataloader) // args.gradient_accumulation_steps // 5 else 1
        args.valid_steps = len(train_dataloader)

        # Prepare optimizer and schedule (linear warmup and decay)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)
        args.warmup_steps = int(t_total * args.warmup_proportion) if args.warmup_steps == 0 else args.warmup_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


        trainer_op = trainer(args=args,
                             model=model,
                             optimizer=optimizer,
                             train_iter=train_dataloader,
                             valid_iter=eval_dataloader,
                             logger=logger,
                             num_epochs=args.num_train_epochs,
                             save_dir=args.output_dir,
                             log_steps=args.logging_steps,
                             valid_steps=args.valid_steps,
                             valid_metric_name="+f1",
                             lr_scheduler=scheduler)
        trainer_op.train()

    # Test
    if args.do_test:
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, logger, data_type='test')
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if "span" not in args.model_type:
            eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
                eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                         collate_fn=collate_fn)
        else:
            eval_dataloader = eval_dataset

        trainer_op = trainer(args=args,
                             model=model,
                             optimizer=None,
                             train_iter=None,
                             valid_iter=None,
                             logger=logger,
                             num_epochs=args.num_train_epochs,
                             save_dir=args.output_dir,
                             log_steps=None,
                             valid_steps=None,
                             valid_metric_name="+f1")

        best_model_file = os.path.join(args.output_dir, "best.model")
        best_train_file = os.path.join(args.output_dir, "best.train")
        trainer_op.load(best_model_file, best_train_file)

        if args.model_type == "bertsoftmax" or args.model_type == "bertcrf":
            metrics = evaluate_bert_normal(args, trainer_op.model, eval_dataloader, logger)
        elif args.model_type == "bertspan":
            metrics = evaluate_bert_span(args, trainer_op.model, eval_dataloader, logger)
        else:
            metrics = evaluate(args, trainer_op.model, eval_dataloader, logger)

    # TODO: Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()