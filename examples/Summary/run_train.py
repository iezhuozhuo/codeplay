# encoding utf-8
import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from train_utils import set_seed, checkoutput_and_setcuda, init_logger #, trainer
from preprocessing import SummaGenCorpus
from ModelConfig import configs

from source.utils.engine import BasicConfig
import source.utils.Constant as constants


def main():
    parser = BasicConfig()
    args = configs(parser)

    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)
    logger.info(args)

    # Set seed
    set_seed(args)

    specials = [constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD]
    processor = SummaGenCorpus(args, specials=specials)
    padding_idx = processor.field["article"].stoi[constants.PAD_WORD]


    # model = model_class(num_filters=args.num_filters,
    #                     embedded_size=args.embedded_size,
    #                     dropout=0.5,
    #                     num_classes=args.num_classes,
    #                     n_vocab=len(processor.vocab),
    #                     filter_sizes=args.filter_sizes,
    #                     embedded_pretrain=embedded_pretrain,
    #                     padding_idx=padding_idx)
    #
    # # Training
    # if args.do_train:
    #     train_dataset = processor.get_train_features()
    #     args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    #     train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #
    #     valid_dataloader = processor.get_dev_features()
    #     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #     eval_sampler = SequentialSampler(valid_dataloader)
    #     eval_dataloader = DataLoader(valid_dataloader, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #     args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #     args.valid_steps = len(train_dataloader)
    #
    #     trainer_op = trainer(args=args,
    #                          model=model,
    #                          optimizer=optimizer,
    #                          train_iter=train_dataloader,
    #                          valid_iter=eval_dataloader,
    #                          logger=logger,
    #                          num_epochs=args.num_train_epochs,
    #                          save_dir=args.output_dir,
    #                          log_steps=args.logging_steps,
    #                          valid_steps=args.valid_steps,
    #                          valid_metric_name="+acc")
    #     trainer_op.train()

    # Test
    if args.do_test:
        pass

    # Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()