import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from preprocessing import classiflyCorpus
from train_utils import textCNNConfig, set_seed, checkoutput_and_setcuda, init_logger, trainer, load_pretrain_embed, evaluate

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

    #数据处理
    corpus = classiflyCorpus(args, embed_file=args.embed_file)
    corpus.load()

    padding_idx = corpus.padding_idx
    embedded_pretrain = None
    if args.embed_file:
        embedded_pretrain = corpus.SRC.embeddings

    model = model_class(num_filters=args.num_filters,
                        embedded_size=args.embedded_size,
                        dropout=0.5,
                        num_classes=args.num_classes,
                        n_vocab=len(corpus.SRC.itos),
                        filter_sizes=args.filter_sizes,
                        embedded_pretrain=embedded_pretrain,
                        padding_idx=padding_idx)
    model = model.to(args.device)
    logger.info(args)

    device = torch.cuda.current_device()
    # Training
    if args.do_train:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = corpus.create_batches(
            args.train_batch_size, "train", shuffle=True, device=device)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = corpus.create_batches(
            args.eval_batch_size, "valid", shuffle=False, device=device)

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.lr_decay is not None and 0 < args.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                            factor=args.lr_decay, patience=1, verbose=True, min_lr=1e-5)
        else:
            lr_scheduler = None

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
                             valid_metric_name="+acc",
                             lr_scheduler = lr_scheduler)
        trainer_op.train()

    # Test
    if args.do_test:
        test_dataloader = corpus.create_batches(
            args.eval_batch_size, "test", shuffle=False, device=device)

        best_model_file = os.path.join(args.output_dir, "best.model")
        best_train_file = os.path.join(args.output_dir, "best.train")
        trainer_op.load(best_model_file, best_train_file)

        metrics = evaluate(args, trainer_op.model, test_dataloader, logger)
        cur_valid_metric = metrics["acc"]
        print(cur_valid_metric)


    # Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()
