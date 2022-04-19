"""Preprocesses entity data into memory mapped file to reduce memory usage by dataloaders."""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabi.config import parser
from tabi.data import EntityDataset
from tabi.utils.train_utils import set_random_seed
from tabi.utils.utils import get_mmap_type, log_setup, save_entity_map, set_device

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger()


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
        tokenizer_name=args.tokenizer_name,
        type_path=args.type_file,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # store tokenized results in memory mapped file
    entity_type = get_mmap_type(args.max_entity_length)
    mmap_file_path = os.path.join(args.log_dir, args.entity_memmap_file)
    logger.info(f"Saving tokenized entity data to {mmap_file_path}.")
    mmap_file = np.memmap(
        mmap_file_path, dtype=entity_type, mode="w+", shape=len(dataset)
    )
    entity_ids = []
    with tqdm(total=len(dataloader), unit="ex", desc="Extracting") as bar:
        for step, batch in enumerate(dataloader):
            entity_batch_data = batch["sample"]
            start_idx = step * args.batch_size
            end_idx = start_idx + args.batch_size
            mmap_file[start_idx:end_idx]["input_ids"] = entity_batch_data["input_ids"]
            mmap_file[start_idx:end_idx]["attention_mask"] = entity_batch_data[
                "attention_mask"
            ]
            mmap_file[start_idx:end_idx]["token_type_ids"] = entity_batch_data[
                "token_type_ids"
            ]
            entity_ids.extend(batch["entity_id"].tolist())
            bar.update(1)
    mmap_file.flush()

    # keep track of embedding idx to entity_id mapping
    id_map_path = os.path.join(args.log_dir, args.entity_map_file)
    save_entity_map(id_map_path, entity_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[parser])
    entity_args = parser.add_argument_group("entity_args")
    entity_args.add_argument("--entity_file", type=str, required=True)
    entity_args.add_argument("--entity_map_file", type=str, default="entity_map.pkl")
    entity_args.add_argument(
        "--entity_memmap_file", type=str, default="entity_data.npy"
    )
    args = parser.parse_args()
    main(args)
