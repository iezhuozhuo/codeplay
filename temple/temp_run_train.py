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
# FIXME 导入config信息
# from ModelConfig import (
#     modelconfig, modelclass
# )
from preprocessing_zh import MatchCorpus
from preprocessing_zh import Example, InputFeatures
from train_utils import evaluate

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger, get_model_parameters_num
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup
from source.modules.embedder import Embedder

# FIXME 加载Class信息
MODEL_CLASSES = {
    # "modelname": (modelconfig, modelclass)
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
    # FIXME 添加 model 注释
    model_log.add_model_remark(remark='本model秀！！！')

    specials = [constants.PAD_WORD, constants.UNK_WORD]
    processor = MatchCorpus(args, specials=specials)
    padding_idx = processor.field["text"].stoi[constants.PAD_WORD]
    args.padding_idx = padding_idx
    # FIXME 是否使用char信息
    # args.num_char_embedding = len(processor.field["char"].stoi)

    embedded_pretrain = Embedder(num_embeddings=processor.field["text"].vocab_size,
                                 embedding_dim=args.embeddin_dim, padding_idx=padding_idx)
    # FIXME 加载预训练model
    # embedded_pretrain.load_embeddingsfor_gensim_vec("/home/gong/zz/data/Match/word2vec.model",
    #                                                 processor.field["text"].stoi)

    logger.info(args)
    model_log.add_param(param_dict=vars(args), param_type='py_param')
    # FIXME 定义model_class读入的参数
    model = model_class(args, embedd=embedded_pretrain)
    model.to(args.device)
    # 打印模型的参数量
    num_parameters = get_model_parameters_num(model)
    logger.info('number of model parameters: {}'.format(num_parameters))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Training
    if args.do_train:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = processor.create_batch(data_type="train")

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = processor.create_batch(data_type="valid")

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader) // args.gradient_accumulation_steps

        # FIXME valid_metric_name
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

    if args.do_test:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_dataloader = processor.create_batch(data_type="test")

        trainer_op = trainer(args=args,
                             model=model,
                             optimizer=optimizer,
                             train_iter=None,
                             valid_iter=None,
                             logger=logger,
                             num_epochs=None,
                             save_dir=args.output_dir,
                             log_steps=None,
                             valid_steps=None,
                             valid_metric_name="+f1")

        best_model_file = os.path.join(args.output_dir, "best.model")
        best_train_file = os.path.join(args.output_dir, "best.train")
        trainer_op.load(best_model_file, best_train_file)

        evaluate(args, trainer_op.model, test_dataloader, logger)


if __name__ == "__main__":
    main()