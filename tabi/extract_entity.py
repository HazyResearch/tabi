"""Extract entity embeddings from a trained biencoder model."""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabi.config import parser
from tabi.data import EntityDataset
from tabi.models.biencoder import Biencoder
from tabi.utils.train_utils import set_random_seed
from tabi.utils.utils import (
    load_model,
    log_setup,
    move_dict,
    save_entity_map,
    set_device,
)

logger = logging.getLogger()

# we only use DataParallel with entity extraction
# to avoid merging back together embeddings across processes with DistributedDataParallel


def main(args):
    # setup log directory and logger
    log_setup(args)

    # set seed and device
    set_random_seed(args.seed)
    set_device(args)

    # datasets and dataloaders
    logger.info("Loading entity dataset...")
    dataset = EntityDataset(
        entity_path=args.entity_file,
        add_entity_type_in_description=args.add_entity_type_in_description,
        max_entity_length=args.max_entity_length,
        tokenized_entity_data=args.tokenized_entity_data,
        tokenizer_name=args.tokenizer_name,
        type_path=args.type_file,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # init model
    model = Biencoder(
        tied=args.tied,
        model_name=args.model_name,
        normalize=args.normalize,
        temperature=args.temperature,
    )
    embed_dim = model.dim

    # load saved model weights
    if args.model_checkpoint is not None:
        load_model(
            model_checkpoint=args.model_checkpoint, device=args.device, model=model
        )

    if args.distributed:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(args.device)

    # save entity embeddings to memory mapped file
    emb_path = os.path.join(args.log_dir, args.entity_emb_path)
    logger.info(f"Saving entity embeddings to {emb_path}.")
    mmap_file = np.memmap(
        emb_path, dtype="float32", mode="w+", shape=(len(dataset), embed_dim)
    )
    model.eval()
    entity_ids = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="ex") as bar:
            for step, batch in enumerate(dataloader):
                entity_emb_batch = model(
                    entity_data=move_dict(batch["sample"], args.device)
                )["entity_embs"]
                start_idx = step * args.batch_size
                end_idx = start_idx + args.batch_size
                mmap_file[start_idx:end_idx] = entity_emb_batch.cpu().numpy()
                entity_ids.extend(batch["entity_id"].tolist())
                bar.update(1)
    mmap_file.flush()

    # keep track of embedding idx to entity_id mapping
    id_map_path = os.path.join(args.log_dir, args.entity_map_file)
    save_entity_map(id_map_path, entity_ids)


if __name__ == "__main__":
    # add arguments specific to entity extraction to parser
    parser = argparse.ArgumentParser(parents=[parser])
    entity_args = parser.add_argument_group("entity_args")
    entity_args.add_argument("--entity_file", type=str, required=True)
    entity_args.add_argument("--model_checkpoint", type=str, required=True)
    entity_args.add_argument("--entity_emb_path", type=str, default="embs.npy")
    entity_args.add_argument("--entity_map_file", type=str, default="entity_map.pkl")
    entity_args.add_argument(
        "--tokenized_entity_data",
        type=str,
        help="File path for memory mapped entity data",
    )
    args = parser.parse_args()
    main(args)
