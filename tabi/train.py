"""Train biencoder model for entity retrieval."""

import argparse
import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tabi.config import parser
from tabi.data import EntityLinkingDataset
from tabi.models.biencoder import Biencoder
from tabi.utils.train_utils import get_bert_optimizer, set_random_seed
from tabi.utils.utils import load_model, log_setup, move_dict, set_device, str2bool

logger = logging.getLogger()


def train(args, model, dataloader, optimizer, tb_writer, epoch, global_step):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_fn = model.module.loss if args.distributed else model.loss
    log_every_k_steps = max(1, int(args.log_freq * len(dataloader)))
    logger.info(f"Logging every {log_every_k_steps} steps")
    with tqdm(total=len(dataloader), unit="ex", desc="Training") as bar:
        for step, batch in enumerate(dataloader):
            start = time.time()
            embs = model(
                entity_data=move_dict(batch["entity"], args.device),
                context_data=move_dict(batch["context"], args.device),
            )
            loss = loss_fn(
                query_embs=embs["query_embs"],
                entity_embs=embs["entity_embs"],
                query_type_labels=batch["query_type_labels"].to(args.device),
                query_entity_labels=batch["query_entity_labels"].to(args.device),
                entity_labels=batch["entity_labels"].to(args.device),
            )
            optimizer.zero_grad()
            loss["total_loss"].backward()
            optimizer.step()
            bar.update(1)
            bar.set_postfix(loss=f'{loss["total_loss"].item():.6f}')
            total_loss += loss["total_loss"].item()
            if (step + 1) % log_every_k_steps == 0:
                logger.info(
                    f"Epoch: {epoch} [{step}/{len(dataloader)}] | loss: {round(loss['total_loss'].item(), 4)} | lr: {optimizer.param_groups[0]['lr']} | {round(time.time()-start, 4)}s/batch"
                )
            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar(
                    "total_loss/train/step",
                    loss["total_loss"].item(),
                    global_step + step,
                )
                tb_writer.add_scalar(
                    "ent_loss/train/step", loss["ent_loss"].item(), global_step + step
                )
                tb_writer.add_scalar(
                    "type_loss/train/step", loss["type_loss"].item(), global_step + step
                )
    avg_loss = total_loss / (step + 1.0)
    return avg_loss


def eval(args, model, dataloader):
    """Eval over entities in batch"""
    model.eval()
    total_loss = 0.0
    loss_fn = model.module.loss if args.distributed else model.loss
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="ex", desc="Evaluating") as bar:
            for step, batch in enumerate(dataloader):
                embs = model(
                    entity_data=move_dict(batch["entity"], args.device),
                    context_data=move_dict(batch["context"], args.device),
                )
                loss = loss_fn(
                    query_embs=embs["query_embs"],
                    entity_embs=embs["entity_embs"],
                    query_type_labels=batch["query_type_labels"].to(args.device),
                    query_entity_labels=batch["query_entity_labels"].to(args.device),
                    entity_labels=batch["entity_labels"].to(args.device),
                )
                bar.update(1)
                bar.set_postfix(loss=f'{loss["total_loss"].item():.6f}')
                total_loss += loss["total_loss"]
    avg_loss = total_loss.item() / (step + 1.0)
    return avg_loss


def main(args):
    # setup log directory and logger
    log_setup(args)

    # setup tensorboard: only first process in distributed logs to tensorboard
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    # set seed and device
    set_random_seed(args.seed)
    set_device(args)

    # prepare datasets and dataloaders
    logger.info("Preparing train dataset...")
    train_dataset = EntityLinkingDataset(
        data_path=args.train_data_file,
        entity_path=args.train_entity_file,
        neg_sample_file=args.train_neg_sample_file,
        num_negatives=args.num_negatives,
        add_entity_type_in_description=args.add_entity_type_in_description,
        max_context_length=args.max_context_length,
        max_entity_length=args.max_entity_length,
        tokenized_entity_data=args.tokenized_entity_data,
        tokenizer_name=args.tokenizer_name,
        type_path=args.type_file,
    )

    logger.info("Preparing dev dataset...")
    dev_dataset = EntityLinkingDataset(
        data_path=args.dev_data_file,
        entity_path=args.dev_entity_file,
        neg_sample_file=args.dev_neg_sample_file,
        num_negatives=args.num_negatives,
        add_entity_type_in_description=args.add_entity_type_in_description,
        max_context_length=args.max_context_length,
        max_entity_length=args.max_entity_length,
        tokenized_entity_data=args.tokenized_entity_data,
        tokenizer_name=args.tokenizer_name,
        type_path=args.type_file,
    )

    # make sure each process only gets its portion of the dataset
    if args.distributed:
        # necessary to set seed here or else seed will be default distributed seed (zero)
        # and data will not change ordering wrt seed argument
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, seed=args.seed, drop_last=True
        )
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset, seed=args.seed, drop_last=True
        )
    else:
        train_sampler = None
        dev_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(dev_sampler is None),
        sampler=dev_sampler,
        drop_last=True,
    )

    # init model
    model = Biencoder(
        tied=args.tied,
        model_name=args.model_name,
        normalize=args.normalize,
        temperature=args.temperature,
        is_distributed=args.local_rank > -1,
        alpha=args.alpha,
    )
    model = model.to(args.device)

    # optimizer
    optimizer = get_bert_optimizer(model, learning_rate=args.lr)

    # lr scheduler
    if args.lr_scheduler_type == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5  # each epoch
        )
    else:
        lr_scheduler = None

    # load saved model weights and load optimizer/scheduler state dicts
    global_step = 0
    starting_epoch = 0
    if args.model_checkpoint is not None:
        model_ckpt_stats = load_model(
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        global_step = model_ckpt_stats["global_step"]
        starting_epoch = model_ckpt_stats["epoch"]

    if lr_scheduler is None:
        for g in optimizer.param_groups:
            g["lr"] = args.lr

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.device], find_unused_parameters=True
        )

    # train loop
    best_dev_loss = float("inf")
    for epoch in range(starting_epoch, starting_epoch + args.n_epochs):
        # required for determinism across runs with distributed
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        avg_train_loss = train(
            args,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            tb_writer=tb_writer,
            epoch=epoch,
            global_step=global_step,
        )
        global_step += len(train_dataloader)
        if args.lr_scheduler_type == "step":
            lr_scheduler.step()

        if lr_scheduler is not None:
            logger.info(
                f"Epoch {epoch} | average train loss: {round(avg_train_loss, 6)} | lr: {lr_scheduler.get_last_lr()[0]}"
            )
        else:
            logger.info(
                f"Epoch {epoch} | average train loss: {round(avg_train_loss, 6)} | lr: {optimizer.param_groups[0]['lr']}"
            )
        logger.info(f"Epoch {epoch} | average train loss: {round(avg_train_loss, 6)}")

        # evaluate on dev set
        avg_dev_loss = eval(args, model=model, dataloader=dev_dataloader)
        logger.info(f"Epoch {epoch} | average dev loss: {round(avg_dev_loss, 6)}")

        # log to tensorboard
        if args.local_rank in [-1, 0]:
            tb_writer.add_scalar("Loss/train/epoch", avg_train_loss, epoch)
            tb_writer.add_scalar("Loss/dev/epoch", avg_dev_loss, epoch)

        # save model at the end of each epoch
        if args.local_rank in [-1, 0]:
            ckpt_path = os.path.join(args.log_dir, f"model_epoch{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "dev_loss": avg_dev_loss,
                    "state_dict": model.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                    if lr_scheduler is not None
                    else None,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_gpu": torch.cuda.get_rng_state()
                    if args.device != "cpu"
                    else None,
                },
                ckpt_path,
            )

        # keep track of best dev score
        if avg_dev_loss < best_dev_loss and args.local_rank in [-1, 0]:
            ckpt_path = os.path.join(args.log_dir, "best_model.pth")
            logger.info(f"Dev loss improved. Saving checkpoint to {ckpt_path}.")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "dev_loss": avg_dev_loss,
                    "state_dict": model.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                    if lr_scheduler is not None
                    else None,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_gpu": torch.cuda.get_rng_state()
                    if args.device != "cpu"
                    else None,
                },
                ckpt_path,
            )
            best_dev_loss = avg_dev_loss

    logger.info("Finished training!")
    if args.local_rank in [-1, 0]:
        # save last ckpt
        last_ckpt_path = os.path.join(args.log_dir, "last_model.pth")
        logger.info(f"Saving last model checkpoint to {last_ckpt_path}.")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "dev_loss": avg_dev_loss,
                "state_dict": model.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
                if lr_scheduler is not None
                else None,
                "rng_cpu": torch.get_rng_state(),
                "rng_gpu": torch.cuda.get_rng_state() if args.device != "cpu" else None,
            },
            last_ckpt_path,
        )

    if args.distributed:
        # tear down the process group
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[parser])
    training_args = parser.add_argument_group("training_args")
    training_args.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    training_args.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="Maximum number of (new) epochs to train (on top of starting epoch).",
    )
    training_args.add_argument(
        "--num_negatives",
        type=int,
        default=5,
        help="Number of hard negative samples to use for training",
    )
    training_args.add_argument(
        "--model_checkpoint", type=str, help="Model checkpoint to continue training"
    )
    training_args.add_argument(
        "--log_freq", type=float, default=0.1, help="Fraction of an epoch to log"
    )
    training_args.add_argument(
        "--alpha", type=float, default=0.1, help="Alpha for weighting type loss"
    )
    training_args.add_argument(
        "--lr_scheduler_type",
        type=str,
        help="LR scheduler: step or leave empty for no LR scheduler",
    )

    data_args = parser.add_argument_group("training_data_args")
    data_args.add_argument("--train_data_file", type=str)
    data_args.add_argument("--dev_data_file", type=str)
    data_args.add_argument("--train_entity_file", type=str)
    data_args.add_argument("--dev_entity_file", type=str)
    data_args.add_argument(
        "--train_neg_sample_file",
        type=str,
        help="File path for negative samples for train dataset",
    )
    data_args.add_argument(
        "--dev_neg_sample_file",
        type=str,
        help="File path for negative samples for dev dataset",
    )
    data_args.add_argument(
        "--tokenized_entity_data",
        type=str,
        help="File path for memory mapped entity data",
    )
    args = parser.parse_args()

    # if using hard negatives, adjust the batch size
    args.orig_batch_size = args.batch_size
    if args.train_neg_sample_file is not None:
        args.batch_size = 2 ** int(
            math.log2(args.batch_size / (args.num_negatives + 1))
        )

    # setup distributed
    args.ngpus_per_node = 1
    if args.distributed:
        dist.init_process_group(backend="nccl")
        logger.info(
            f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
            end="",
        )

        # update batch size and number of workers for DistributedDataParallel
        # assumes we are using a single GPU per process
        args.ngpus_per_node = torch.cuda.device_count()
        args.batch_size = args.batch_size // args.ngpus_per_node

    main(args)
