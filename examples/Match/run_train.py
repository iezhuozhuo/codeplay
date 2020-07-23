# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

import os
import random
import numpy as np

from torch.optim import Adam

from train_utils import trainer
from ModelConfig import (
    ARCIConfig, ARCIModel,
    ARCIIConfig, ARCIIModel,
    MVLSTMConfig, MVLSTModel,
    MatchPyramidConig, MatchPyramidModel, BiMPM)
from preprocessing import MatchCorpus

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup
from source.modules.embedder import Embedder


MODEL_CLASSES = {
    "arci": (ARCIConfig, ARCIModel),
    "arcii": (ARCIIConfig, ARCIIModel),
    "mv_lstm": (MVLSTMConfig, MVLSTModel),
    "matchpyramid": (MatchPyramidConig, MatchPyramidModel),
    "bimpm": (MVLSTMConfig, BiMPM)
}


def main():
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    configs, model_class = MODEL_CLASSES[model_type]
    args = configs(parser)

    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)

    # Set seed
    set_seed(args)

    specials = [constants.PAD_WORD, constants.UNK_WORD]
    processor = MatchCorpus(args, specials=specials)
    padding_idx = processor.field["text"].stoi[constants.PAD_WORD]

    embedded_pretrain = Embedder(num_embeddings=processor.field["text"].vocab_size,
                                 embedding_dim=128, padding_idx=padding_idx)
    # embedded_pretrain.load_embeddingsfor_gensim_vec("/home/administrator4/codeplay/source/utils/word2vec.model",
    #                                                 processor.field["text"].stoi)

    logger.info(args)

    model = model_class(args, embedded_pretrain, processor.field["text"].vocab_size, padding_idx)
    model.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

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
                             valid_metric_name="-loss",
                             processor=processor)
        trainer_op.train()

if __name__ == "__main__":
    main()