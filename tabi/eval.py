"""Retrieve candidates for evaluation or hard negative sampling."""

import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabi.config import parser
from tabi.data import EntityLinkingDataset
from tabi.models.biencoder import Biencoder
from tabi.utils.train_utils import set_random_seed
from tabi.utils.utils import (
    combine_negs,
    combine_preds,
    load_model,
    log_setup,
    move_dict,
    set_device,
    str2bool,
    write_neg_samples,
    write_preds,
)

logger = logging.getLogger()


def main(args):
    # setup log directory and logger
    log_setup(args)

    # set seed and device
    set_random_seed(args.seed)
    set_device(args)

    # datasets and dataloaders
    logger.info("Preparing dataset...")

    top_k = args.top_k if args.mode == "eval" else args.orig_num_negatives
    dataset = EntityLinkingDataset(
        data_path=args.test_data_file,
        entity_path=args.entity_file,
        top_k=top_k,
        tokenize_entities=False,
        max_context_length=args.max_context_length,
        max_entity_length=args.max_entity_length,
        tokenizer_name=args.tokenizer_name,
        type_path=args.type_file,
        is_eval=args.mode == "eval",
    )

    # make sure each process only gets its portion of the dataset
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=sampler,
    )

    # init model
    model = Biencoder(
        tied=args.tied,
        entity_emb_path=args.entity_emb_path,
        top_k=top_k,
        model_name=args.model_name,
        normalize=args.normalize,
        temperature=args.temperature,
    )

    # load saved model weights
    if args.model_checkpoint is not None:
        load_model(
            model_checkpoint=args.model_checkpoint, device=args.device, model=model
        )

    model = model.to(args.device)
    # entity embs aren't parameters of the model so need to be moved separately
    model.entity_embs = model.entity_embs.to(args.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.device], find_unused_parameters=True
        )

    # run evaluation
    model.eval()
    predictions = []
    predict_fn = model.module.predict if args.distributed else model.predict
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="ex", desc="Running model") as bar:
            for batch in dataloader:
                prediction = predict_fn(
                    context_data=move_dict(batch["context"], args.device),
                    data_id=batch["data_id"],
                )
                predictions.append(prediction)
                bar.update(1)

    if args.mode == "eval":
        # save predictions as jsonlines
        pred_file = os.path.join(
            args.log_dir, args.pred_file.split(".jsonl")[0] + f"_{args.device if args.device != 'cpu' else 0}.jsonl"
        )
        write_preds(
            preds=predictions,
            dataset=dataset,
            pred_file=pred_file,
            entity_map_file=args.entity_map_file,
        )
        # make sure all processes write their predictions
        if args.distributed:
            torch.distributed.barrier()
        # let first process combine predictions across processes into a single file
        if args.local_rank in [-1, 0]:
            combine_preds(
                args.log_dir,
                os.path.join(args.log_dir, args.pred_file),
                num_gpus=dist.get_world_size() if args.distributed else 1,
            )

    elif args.mode == "neg_sample":
        # save negative samples to file
        neg_sample_file = os.path.join(
            args.log_dir,
            args.neg_sample_file.split(".json")[0] + f"_{args.device if args.device != 'cpu' else 0}.json",
        )
        write_neg_samples(
            preds=predictions,
            dataset=dataset,
            entity_map_file=args.entity_map_file,
            neg_sample_file=neg_sample_file,
            num_negative_samples=args.orig_num_negatives,
        )
        if args.distributed:
            # wait for all devices to generate their negative samples
            torch.distributed.barrier()
        # let first process combine negative samples across processes into a single file
        if args.local_rank in [-1, 0]:
            if args.filter_negatives:
                combine_negs(
                    log_dir=args.log_dir,
                    neg_sample_file=os.path.join(args.log_dir, args.neg_sample_file),
                    num_gpus=dist.get_world_size() if args.distributed else 1,
                    use_filter=args.filter_negatives,
                    entity_cache=dataset.entity_cache,
                    ent_counter=dataset.ent_counter,
                    # top_k here is the number of negatives we want after filter
                    total_num_samples=args.top_k,
                    mult_factor=args.mult_factor,
                )
            else:
                combine_negs(
                    log_dir=args.log_dir,
                    neg_sample_file=os.path.join(args.log_dir, args.neg_sample_file),
                    num_gpus=dist.get_world_size() if args.distributed else 1,
                    use_filter=args.filter_negatives,
                )

    if args.distributed:
        # TODO: automatically remove all extra negative sample files that are generated
        torch.distributed.barrier()
        # tear down the process group
        dist.destroy_process_group()


if __name__ == "__main__":
    # add arguments specific to eval to parser
    parser = argparse.ArgumentParser(parents=[parser])
    eval_args = parser.add_argument_group("eval_args")
    eval_args.add_argument("--test_data_file", type=str, required=True)
    eval_args.add_argument("--entity_file", type=str, required=True)
    eval_args.add_argument("--model_checkpoint", type=str, required=True)
    eval_args.add_argument("--entity_emb_path", type=str, required=True)
    eval_args.add_argument("--entity_map_file", type=str)
    eval_args.add_argument(
        "--mode", type=str, default="eval", choices=["eval", "neg_sample"]
    )
    eval_args.add_argument("--pred_file", type=str, default="preds.jsonl")
    eval_args.add_argument("--neg_sample_file", type=str, default="neg_samples.json")
    eval_args.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of candidates to retrieve for each mention",
    )
    eval_args.add_argument(
        "--filter_negatives",
        type=str2bool,
        default=False,
        help="Whether to filter negatives by entity count",
    )
    # eval_args.add_argument("--popularity_file", type=str, help="File with entity counts for filtering negatives")
    eval_args.add_argument(
        "--mult_factor",
        type=int,
        default=10,
        help="Multiplicative factor for ratio of neg:pos in negative sample filter, e.g. up to 10 negatives for every 1 positive.",
    )
    eval_args.add_argument(
        "--orig_num_negatives",
        type=int,
        default=20,
        help="Number of negatives to fetch for filtering",
    )
    args = parser.parse_args()

    # setup distributed

    # recommend using distributed for neg_sample mode but not for eval mode
    # for distributed eval, the accuracy metrics are computed separately over each portion of the dataset
    if args.distributed:
        dist.init_process_group(backend="nccl")
        logger.info(
            f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
            end="",
        )

        # update batch size and number of workers for DistributedDataParallel
        # assumes we are using a single GPU per process
        ngpus_per_node = torch.cuda.device_count()
        args.batch_size = args.batch_size // ngpus_per_node

    main(args)
