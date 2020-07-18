# encoding utf-8
import os
import random
import numpy as np

from torch.optim import adam

from train_utils import trainer
from preprocessing import SummaGenCorpus
from ModelConfig import configs, PointNetworkModel
from Metrics import BeamSearch

from source.utils.engine import BasicConfig
import source.utils.Constant as constants
from source.callback.optimizater.optimCustom import AdagradCustom
from source.utils.misc import set_seed, checkoutput_and_setcuda, init_logger
from source.callback.optimizater.adamw import AdamW
from source.callback.lr_scheduler import get_linear_schedule_with_warmup

MODEL_CLASSES = {
    "summagen": (PointNetworkModel, configs)
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

    specials = [constants.PAD_WORD, constants.UNK_WORD, constants.BOS_WORD, constants.EOS_WORD]
    processor = SummaGenCorpus(args, specials=specials)
    padding_idx = processor.field["article"].stoi[constants.PAD_WORD]

    embedded_pretrain = None

    logger.info(args)

    model = model_class(args, embedded_pretrain, processor.field["article"].vocab_size, padding_idx)
    model.to(args.device)

    initial_lr = args.learning_rate
    adagrad_init_acc = 0.1
    params = list(model.parameters())
    optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=adagrad_init_acc)

    # Training
    if args.do_train:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader = processor.create_batch(data_type="train")

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = processor.create_batch(data_type="valid")

        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps // 5
        args.valid_steps = len(train_dataloader)

        # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      "weight_decay": args.weight_decay, },
        #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0},
        # ]
        # args.warmup_steps = int(t_total * args.warmup_proportion) if args.warmup_steps == 0 else args.warmup_steps
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        #                                             num_training_steps=t_total)

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
                             lr_scheduler=None,
                             grad_clip=5.0)
        trainer_op.train()

    # Test
    if args.do_test:
        # batch_size = 1
        test_dataloader, origin_dataset = processor.create_batch(data_type="test")
        args.is_coverage = True
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
                             valid_metric_name="-loss")

        best_model_file = os.path.join(args.output_dir, "best.model")
        best_train_file = os.path.join(args.output_dir, "best.train")
        trainer_op.load(best_model_file, best_train_file)

        BS = BeamSearch(args=args,
                        model=trainer_op.model,
                        test_dataset=test_dataloader,
                        origin_dataset=origin_dataset,
                        vocab=processor.field["article"],
                        logger=logger)

        BS.decode()

    # Infer case study
    if args.do_infer:
        pass


if __name__ == "__main__":
    main()