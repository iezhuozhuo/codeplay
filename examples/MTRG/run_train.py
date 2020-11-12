import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import WEIGHTS_NAME, BertTokenizer
from transformers.modeling_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel

from preprocessing import load_and_cache_examples_gpt, collate_fn_gpt, GPTProcessor
from train_utils import (trainer, evaluate,)
from ModelConfig import GPT2ModelConfig

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger, get_model_parameters_num

MODEL_CLASSES = {
    'gpt2': (GPT2ModelConfig, GPT2LMHeadModel, BertTokenizer),

}

ProcessorClass = {
    "gpt2": GPTProcessor,
}


def main():
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    configs, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    args, config_class = configs(parser)  # 返回的是 GPT2Config
    if args.device_name != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)

    # Set seed
    set_seed(args)

    # specials = [constants.UNK_WORD, constants.PAD_WORD, constants.CLS, constants.SEP]
    processor = ProcessorClass[args.model_type]()

    # 初始化tokenizer
    tokenizer = tokenizer_class(vocab_file=args.tokenizer_name_or_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids(constants.PAD)
    print(pad_id)
    if args.model_name_or_path:  # 如果指定了预训练的GPT2模型
        model = model_class.from_pretrained(args.model_name_or_path)
    else:  # 若没有指定预训练模型，则初始化模型
        model_config = config_class.from_json_file(args.config_name if args.config_name else args.model_name_or_path)
        model = GPT2LMHeadModel(config=model_config)
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    model.resize_token_embeddings(vocab_size)
    args.max_seq_length = model.config.to_dict().get("n_ctx")

    model.to(args.device)

    num_parameters = get_model_parameters_num(model)
    logger.info('number of model parameters: {}'.format(num_parameters))

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples_gpt(args, processor, tokenizer, logger, data_type='train')
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        #                               collate_fn=collate_fn_gpt)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.num_workers,
                                      collate_fn=collate_fn_gpt)

        eval_dataset = load_and_cache_examples_gpt(args, processor, tokenizer, logger, data_type='dev')
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate_fn_gpt)

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5 if len(
            train_dataloader) // args.gradient_accumulation_steps // 5 else 1
        args.valid_steps = len(train_dataloader) // args.gradient_accumulation_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        optimizer_grouped_parameters = model.parameters()
        args.warmup_steps = int(t_total * args.warmup_proportion) if args.warmup_steps == 0 else args.warmup_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=True, eps=args.adam_epsilon)
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
                             valid_metric_name="-loss",
                             lr_scheduler=scheduler)
        trainer_op.train()

    # Test
    if args.do_test:
        pass
        # eval_dataset = load_and_cache_examples_gpt(args, processor, tokenizer, logger, data_type='test')
        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # if "span" not in args.model_type:
        #     eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
        #         eval_dataset)
        #     eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
        #                                  collate_fn=collate_fn_gpt)
        # else:
        #     eval_dataloader = eval_dataset
        #
        # trainer_op = trainer(args=args,
        #                      model=model,
        #                      optimizer=None,
        #                      train_iter=None,
        #                      valid_iter=None,
        #                      logger=logger,
        #                      num_epochs=args.num_train_epochs,
        #                      save_dir=args.output_dir,
        #                      log_steps=None,
        #                      valid_steps=None,
        #                      valid_metric_name="+f1")
        #
        # best_model_file = os.path.join(args.output_dir, "best.model")
        # best_train_file = os.path.join(args.output_dir, "best.train")
        # trainer_op.load(best_model_file, best_train_file)

    # TODO: Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()
    # from transformers import AutoTokenizer, AutoModelWithLMHead
    #
    # # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    # config = BertConfig.from_pretrained("bert-base-chinese")
    # # print(config)
