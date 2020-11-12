import os
import random
import numpy as np
# from sklearn import preprocessing

import torch
import pickle
from preprocessing import UbuntuCorpus,MTRSFeatures
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger
from train_utils import trainer, evaluate
from ModelConfig import (
            SMNConfig,
            SMNModel)

from source.utils.engine import BasicConfig


MODEL_CLASSES = {
    "smn":(SMNModel,SMNConfig)
}

def main():
    
    parser = BasicConfig()
    model_type = vars(parser.parse_known_args()[0])["model_type"].lower()
    model_class, configs = MODEL_CLASSES[model_type]
    args = configs(parser)
    args = checkoutput_and_setcuda(args)
    logger = init_logger(args)
    logger.info('Dataset collected from {}'.format(args.data_dir))
    # Set seed
    set_seed(args)
    processor = UbuntuCorpus(args)

    logger.info(args)

    pretrained_weight = pickle.load(open(args.data_dir + 'embedding.pkl', 'rb'), encoding="bytes")
    pretrained_weight = np.array(pretrained_weight)
    model = model_class(args=args, embedded_pretrain=pretrained_weight)
 
    # model.to(args.device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    if args.do_train:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = processor.create_batch(data_type="train")

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = processor.create_batch(data_type="eval")

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader)// args.gradient_accumulation_steps
        
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
                             valid_metric_name="+R10@1")
        trainer_op.train()
    print('training complete!')
    # Test
    if args.do_test:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        test_dataloader = processor.create_batch(data_type="eval")

        trainer_op = trainer(args=args,
                             model=model,
                             optimizer=optimizer,
                             train_iter=None,
                             eval_iter=None,
                             logger=logger,
                             num_epochs=args.num_train_epochs,
                             save_dir=args.output_dir,
                             log_steps=None,
                             valid_steps=None,
                             valid_metric_name="+R10@1")

        best_model_file = os.path.join(args.output_dir, args.fusion_type+"_best.model")
        best_train_file = os.path.join(args.output_dir, args.fusion_type+"_best.train")

        trainer_op.load(best_model_file, best_train_file)

        evaluate(args, trainer_op.model, test_dataloader, logger)
    print('test complete')
    # TODO: Infer case study
    if args.do_infer:
        #有点麻烦,懒得写了
        pass


if __name__ == "__main__":
    main()