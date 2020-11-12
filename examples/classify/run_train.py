import os
import random
import numpy as np

import torch

from preprocessing_zh import DataProcessor, ClassifierCorpus
from train_utils import set_seed, checkoutput_and_setcuda, init_logger, trainer, load_pretrain_embed, evaluate
from ModelConfig import (
            TextCNNConfig,
            TextRNNConfig,
            DPCNNConfig,
            TransformerClassifierConfig,
            FastTextConfig,
            TextRNNModel,
            TextCNNModel,
            DPCNNModel,
            TransformerClassifierModel,
            FastTextModel)

from source.utils.engine import BasicConfig
import source.utils.Constant as constants

MODEL_CLASSES = {
    "textcnn": (TextCNNModel, TextCNNConfig),
    "textrnn": (TextRNNModel, TextRNNConfig),
    "dpcnn": (DPCNNModel, DPCNNConfig),
    "transformer": (TransformerClassifierModel, TransformerClassifierConfig),
    "fasttext": (FastTextModel, FastTextConfig)
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

    specials = [constants.UNK_WORD, constants.PAD_WORD]
    processor = ClassifierCorpus(args=args, specials=specials)
    padding_idx = processor.field["text"].stoi[constants.PAD_WORD]
    embedded_pretrain = None
    if args.embed_file:
        embedded_pretrain = load_pretrain_embed(args.embed_file, args.output_dir, processor.field["text"].stoi, args.embedded_size)

    logger.info(args)

    model = model_class(args=args,
                        embedded_pretrain=embedded_pretrain,
                        n_vocab=processor.field["text"].vocab_size,
                        padding_idx=padding_idx)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    if args.do_train:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = processor.create_batch(data_type="train")

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = processor.create_batch(data_type="valid")

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader)

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
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_dataloader = processor.create_batch(data_type="test")

        trainer_op = trainer(args=args,
                             model=model,
                             optimizer=optimizer,
                             train_iter=None,
                             valid_iter=None,
                             logger=logger,
                             num_epochs=args.num_train_epochs,
                             save_dir=args.output_dir,
                             log_steps=None,
                             valid_steps=None,
                             valid_metric_name="+acc")

        best_model_file = os.path.join(args.output_dir, "best.model")
        best_train_file = os.path.join(args.output_dir, "best.train")
        trainer_op.load(best_model_file, best_train_file)

        evaluate(args, trainer_op.model, test_dataloader, logger)

    # TODO: Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()

