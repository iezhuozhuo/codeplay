from dataloader import Vocab, DataLoaders, convert_examples_to_features
import numpy as np
import torch
import random
import argparse
import logging
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
logger = logging.getLogger(__name__)


def load_and_cache_examples(args, filename, vocab_src, vocab_tgt, for_train, with_S=False, with_E=False):
    logger.info("Creating features from dataset file at %s", filename)
    processor = DataLoaders(for_train)
    examples = processor.get_train_data(filename)
    dataset = convert_examples_to_features(args, examples, vocab_src, vocab_tgt, with_S=with_S, with_E=with_E)
    return dataset


def train(args, model, vocab_src, vocab_tgt, for_train, with_S=False, with_E=False):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, args.train_data, vocab_src, vocab_tgt, for_train, with_S=with_S, with_E=with_E)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_epoch = 0
    loss_accumulated, acc_accumulated = 0., 0.
    model.zero_grad()
    for ep in range(1, int(args.num_train_epochs) + 1):
        logger.info("Epoch - {}".format(ep))
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            loss, acc = model(batch[0], batch[1])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            loss_accumulated += loss.item()
            acc_accumulated += acc
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            model.zero_grad()
            global_step += 1

            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    dev_acc = evaluate(args, model, vocab_src, vocab_tgt, for_train, with_S=with_S, with_E=with_E)
                    print("Dev Batch, acc %.5f" % (dev_acc))
                if best_dev_acc < dev_acc:
                    best_model_save_path = os.path.join(args.output_dir, "best_model/")
                    if not os.path.exists(best_model_save_path):
                        os.mkdir(best_model_save_path)
                    best_epoch = ep
                    logger.info("Saving model checkpoint to %s", best_model_save_path)
                    best_dev_acc = dev_acc
                    best_model_save_file = os.path.join(best_model_save_path, 'best_model')
                    torch.save({'args': args, 'model': model.state_dict()}, best_model_save_file)


def evaluate(args, model, vocab_src, vocab_tgt, for_train, with_S=False, with_E=False):
    dataset = load_and_cache_examples(args, args.dev_data, vocab_src, vocab_tgt, for_train, with_S=with_S, with_E=with_E)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    dev_acc = 0.

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            src_input, tgt_input = batch[0], batch[1]
            _, acc = model(src_input, tgt_input)
            dev_acc += np.sum(acc.cpu().numpy())
    dev_acc = dev_acc / len(eval_dataloader)
    return dev_acc


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_src', type=str)
    parser.add_argument('--vocab_tgt', type=str)
    parser.add_argument('--train_data', type=str, default="")
    parser.add_argument('--dev_data', type=str, default="")
    parser.add_argument("--output_dir", type=str, default="../output")

    parser.add_argument('--which_ranker', type=str, default="ranker")
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--ff_embed_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=30)

    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=16)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=16)

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    return parser.parse_known_args()


def update_lr(optimizer, coefficient):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * coefficient


def main():
    random.seed(19940117)
    torch.manual_seed(19940117)
    args, _ = parse_config()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    if args.which_ranker == 'ranker':
        from ranker import Ranker
    elif args.which_ranker == 'masker_ranker':
        from masker_ranker import Ranker

    vocab_src = Vocab(args.vocab_src, with_SE=False)
    vocab_tgt = Vocab(args.vocab_tgt, with_SE=False)
    logging.info("Load Vocab finished")
    model = Ranker(args, vocab_src, vocab_tgt,
            args.embed_dim, args.ff_embed_dim,
            args.num_heads, args.dropout, args.num_layers)
    model = model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train(args, model, vocab_src, vocab_tgt, for_train=True, with_S=False, with_E=False)


if __name__ == "__main__":
    main()