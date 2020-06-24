import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from preprocessing import DataProcessor
from train_utils import textCNNConfig, set_seed, checkoutput_and_setcuda, init_logger, trainer, load_pretrain_embed

from source.models.TextCNN import TextCNN
from source.utils.engine import BasicConfig
import source.utils.Constant as constants

MODEL_CLASSES = {
    "textcnn": (TextCNN, textCNNConfig),
}


def main():
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    model_class, configs = MODEL_CLASSES[model_type]
    args = configs(parser)

    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)

    # Set seed
    set_seed(args)

    processor = DataProcessor(args)
    padding_idx = processor.vocab[constants.PAD_WORD]
    embedded_pretrain = None
    if args.embed_file:
        embedded_pretrain = load_pretrain_embed(args.embed_file, args.output_dir, processor.vocab, args.embedded_size)

    model = model_class(num_filters=args.num_filters,
                        embedded_size=args.embedded_size,
                        dropout=0.5,
                        num_classes=args.num_classes,
                        n_vocab=len(processor.vocab),
                        filter_sizes=args.filter_sizes,
                        embedded_pretrain=embedded_pretrain,
                        padding_idx=padding_idx)

    logger.info(args)
    # Training
    if args.do_train:
        train_dataset = processor.get_train_features()
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        valid_dataloader = processor.get_dev_features()
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(valid_dataloader)
        eval_dataloader = DataLoader(valid_dataloader, sampler=eval_sampler, batch_size=args.eval_batch_size)

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
                             valid_metric_name="+acc")
        trainer_op.train()

    # Test
    if args.do_test:
        pass

    # Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()
