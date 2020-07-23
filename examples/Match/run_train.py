# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

import os
import random
import numpy as np
from model_log.modellog import ModelLog

from torch.optim import Adam

from train_utils import trainer
from ModelConfig import (
    ARCIConfig, ARCIModel,
    ARCIIConfig, ARCIIModel,
    MVLSTMConfig, MVLSTMoel,
    MatchPyramidConig, MatchPyramidModel,
    MwANConfig, MwANModel,
    BiMPMConfig, BiMPMModule,
    ESIMConfig, ESIMModel
)
from preprocessing import MatchCorpus
from preprocessing import Example, InputFeatures

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup
from source.modules.embedder import Embedder


MODEL_CLASSES = {
    "arci": (ARCIConfig, ARCIModel),
    "arcii": (ARCIIConfig, ARCIIModel),
    "mvlstm": (MVLSTMConfig, MVLSTMoel),
    "matchpyramid": (MatchPyramidConig, MatchPyramidModel),
    "mwan": (MwANConfig, MwANModel),
    "bimpm": (BiMPMConfig, BiMPMModule),
    "esim": (ESIMConfig, ESIMModel)
}


def main():
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    configs, model_class = MODEL_CLASSES[model_type]
    args = configs(parser)

    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)
    logger.info("Load {} model".format(args.model_type))
    # Set seed
    set_seed(args)

    model_log = ModelLog(nick_name='zhuo', project_name='DeepMatch', project_remark='')
    model_log.add_model_name(model_name=args.model_type.upper())
    if args.aug:
        model_log.add_model_remark(remark='使用aug')
    else:
        model_log.add_model_remark(remark='不使用aug')

    specials = [constants.PAD_WORD, constants.UNK_WORD]
    processor = MatchCorpus(args, specials=specials)
    padding_idx = processor.field["text"].stoi[constants.PAD_WORD]
    args.padding_idx = padding_idx
    embedded_pretrain = Embedder(num_embeddings=processor.field["text"].vocab_size,
                                 embedding_dim=128, padding_idx=padding_idx)
    embedded_pretrain.load_embeddingsfor_gensim_vec("/home/gong/zz/data/Match/word2vec.model",
                                                    processor.field["text"].stoi)

    logger.info(args)
    model_log.add_param(param_dict=vars(args), param_type='py_param')
    model = model_class(args, embedd=embedded_pretrain)
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
                             valid_metric_name="+f1",
                             model_log=model_log)
        trainer_op.train()


if __name__ == "__main__":
    main()
