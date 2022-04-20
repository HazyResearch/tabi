import logging
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict

import jsonlines
import numpy as np
import torch
import ujson
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_entity_map(id_map_path, entity_ids):
    logger.info(f"Saving ids to {id_map_path}.")
    entity_map = {int(idx): int(eid) for idx, eid in enumerate(entity_ids)}
    with open(id_map_path, "wb") as f:
        pickle.dump(entity_map, f)


def get_type_str(types):
    all_types = []
    for t in types:
        for subt in t.split("/")[1:]:
            if subt not in all_types:
                all_types.append(subt)
    return " ".join(all_types)


def filter_negatives(
    negative_samples, entity_cache, ent_counter, mult_factor=1, total_num_samples=1
):
    neg_counter = Counter()
    missing_negs = 0
    # sample random ids in advance -- sample max possible number to avoid running out of random ids
    start = time.time()
    rand_samples = list(
        np.random.choice(
            list(entity_cache.keys()), len(negative_samples) * total_num_samples
        )
    )
    logger.info(f"Time to sample random ids: {round(time.time()-start,3)}s")
    for row, val in tqdm(negative_samples.items(), desc="Filtering"):
        new_samples = []
        new_scores = []
        samples = val["samples"]
        scores = val["scores"]
        for sample, score in zip(samples, scores):
            pos_count = ent_counter[sample]
            if neg_counter[sample] < pos_count * mult_factor:
                new_samples.append(sample)
                new_scores.append(score)
                neg_counter[sample] += 1
            # exit if we have all the samples we need so the remaining
            # hard samples can be used for another example
            if len(new_samples) == total_num_samples:
                break
        while len(new_samples) < total_num_samples:
            missing_negs += 1
            new_samples.append(int(rand_samples.pop()))
        negative_samples[row]["samples"] = new_samples[:total_num_samples]
        negative_samples[row]["scores"] = new_scores[:total_num_samples]
    logger.info(
        f"{round(missing_negs/(len(negative_samples)*total_num_samples)*100,3)}% random samples"
    )
    return negative_samples


def get_mmap_type(max_length: int):
    """Get datatype for storing tokenized data in memory mapped file.

    Modified from https://github.com/senwu/Emmental-Candidate_retrieval/blob/master/data_processing/prep_entity_mmap.py
    """
    return [
        ("input_ids", "i8", max_length),
        ("attention_mask", "i8", max_length),
        ("token_type_ids", "i8", max_length),
    ]


def load_model(model_checkpoint, model, device, optimizer=None, lr_scheduler=None):
    """Load model checkpoint and update optimizer and lr_scheduler if in checkpoint.

    Returns:
        dict with global_step and epoch to start training from
    """
    logger.info(f"Loading model from checkpoint {model_checkpoint}")
    if device != "cpu":
        # map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(device)
        checkpoint = torch.load(model_checkpoint, map_location=loc)
    else:
        checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))
    # remove DDP "module." prefix if present
    state_dict = checkpoint["state_dict"]
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Loaded optimizer.")

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info("Loaded lr scheduler.")

    # reload random states to resume data loading order across runs
    # TODO: support reloading numpy state if any new randomness depends on numpy random
    if "rng_cpu" in checkpoint:
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if device != "cpu":
            torch.cuda.set_rng_state(checkpoint["rng_gpu"].cpu())
        logger.debug("Loaded random states.")

    return {
        "global_step": checkpoint.get("global_step", 0),
        "epoch": checkpoint.get("epoch", -1) + 1,
    }


def set_device(args):
    if args.distributed:
        # set to device 0 if using data parallel
        args.device = args.local_rank if args.local_rank != -1 else 0
    else:
        args.device = args.gpu if args.gpu >= 0 else "cpu"
    if args.device != "cpu":
        torch.cuda.set_device(args.device)
    logger.info(f"Device id: {args.device}")


def move_dict(dict_to_move, device):
    return {k: val.to(device) for k, val in dict_to_move.items()}


def combine_preds(log_dir, predfile, num_gpus):
    output = jsonlines.open(predfile, "w")
    logger.info(f"Writing final preds to {predfile}")
    seen_id = set()
    for gpu_idx in range(num_gpus):
        with jsonlines.open(
            f'{log_dir}/{os.path.basename(predfile).split(".jsonl")[0]}_{gpu_idx}.jsonl'
        ) as f:
            for line in f:
                line_id = line["id"]
                # already seen in another process
                if line_id in seen_id:
                    continue
                seen_id.add(line_id)
                output.write(line)
    output.close()


def combine_negs(
    log_dir,
    neg_sample_file,
    num_gpus,
    use_filter=False,
    entity_cache=None,
    ent_counter=None,
    total_num_samples=-1,
    mult_factor=1,
):
    neg_samples = {}
    for gpu_idx in range(num_gpus):
        # includes sample ids and distances
        with open(f"{log_dir}/neg_samples_{gpu_idx}.json", "r") as f:
            res = ujson.load(f)
            neg_samples.update(res)

    # filter negatives before saving combined negatives
    if use_filter:
        neg_samples = filter_negatives(
            neg_samples,
            entity_cache=entity_cache,
            ent_counter=ent_counter,
            # top_k here is the number of negatives we want after filter
            total_num_samples=total_num_samples,
            mult_factor=mult_factor,
        )

    with open(neg_sample_file, "w") as f:
        ujson.dump(neg_samples, f)

    logger.info(f"Wrote final negative samples to {neg_sample_file}")


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def write_neg_samples(
    preds, dataset, entity_map_file, neg_sample_file, num_negative_samples
):
    # keep track of emb id to eid
    with open(entity_map_file, "rb") as f:
        entity_map = pickle.load(f)

    # flatten preds across batches
    flattened_preds = flatten_dicts(preds)

    # store as dictionary because indices will be accessed out of order
    # during training
    neg_samples = {}
    # need to use data_id rather than idx in list to support ddp
    for (data_id, indices, scores) in zip(
        flattened_preds["data_id"],
        flattened_preds["indices"],
        flattened_preds["scores"],
    ):

        neg_samples_ex = []
        scores_ex = []
        gold_eids = [dataset.data[data_id]["gold"]]
        gold_eids.extend(dataset.data[data_id]["alt_gold"])
        gold_eids = set(gold_eids)
        for s, idx in zip(scores, indices):
            eid = entity_map[idx]
            # remove gold entities from hard negatives
            if eid in gold_eids:
                continue
            scores_ex.append(float(s))
            # save entity ids (not emb/row ids) for negative samples
            neg_samples_ex.append(eid)
        neg_samples[int(data_id)] = {
            "samples": neg_samples_ex[:num_negative_samples],
            "scores": scores_ex[:num_negative_samples],
        }

    with open(neg_sample_file, "w") as f:
        ujson.dump(neg_samples, f)

    logger.info(f"Wrote negative samples to {neg_sample_file}")


def correct_at_k(pred_ids, gold_ids, k):
    """Return 1 if *any* gold id occurs in the top-k predicted ids, else 0."""
    return int(len(set(gold_ids).intersection(pred_ids[:k])) > 0)


def write_preds(preds, dataset, pred_file, entity_map_file):
    # embedding id to eid
    entity_map = None
    if entity_map_file is not None:
        with open(entity_map_file, "rb") as f:
            entity_map = pickle.load(f)
    entity_cache = dataset.entity_cache

    # flatten preds across batches
    flattened_preds = flatten_dicts(preds)
    # KILT FORMAT
    # {
    # "id": x,
    # "output": [
    # 	{
    # 	"answer": y,
    # 	"provenance": [
    # 		{"wikipedia_id": z},
    # 		{"wikipedia_id": w},
    # 		...
    # 	]
    # 	}
    # ]
    # }
    total_at_1 = 0
    total_at_10 = 0
    iter_ = tqdm(range(len(flattened_preds["data_id"])), desc="Evaluating")
    with jsonlines.open(pred_file, "w") as f_out:
        for i in iter_:
            if entity_map is not None:
                pred_eids = [
                    entity_map[emb_id] for emb_id in flattened_preds["indices"][i]
                ]
            else:
                pred_eids = flattened_preds["indices"][i]
            data_id = flattened_preds["data_id"][i]
            orig_data_id = dataset.data[data_id]["id"]
            try:
                new_ids = [
                    {
                        "wikipedia_id": entity_cache[eid]["wikipedia_page_id"],
                        "wikipedia_title": entity_cache[eid]["title"],
                    }
                    for eid in pred_eids
                ]
            except:
                # not using wikipedia ids
                new_ids = [
                    {
                        "kb_id": eid,
                        "title": entity_cache[eid]["title"],
                    }
                    for eid in pred_eids
                ]
            gold_ids = [dataset.data[data_id]["gold"]]
            gold_ids.extend(dataset.data[data_id]["alt_gold"])
            total_at_1 += correct_at_k(pred_ids=pred_eids, gold_ids=gold_ids, k=1)
            total_at_10 += correct_at_k(pred_ids=pred_eids, gold_ids=gold_ids, k=10)
            iter_.set_postfix(acc_1=total_at_1 / (i + 1), acc_10=total_at_10 / (i + 1))
            output = {
                "id": orig_data_id,
                "input": dataset.data[data_id]["text"],
                "output": [{"answer": "", "provenance": new_ids}],
            }
            f_out.write(output)
    logger.info(f"Accuracy@1: {round(total_at_1/(i+1), 5)}")
    logger.info(f"Accuracy@10: {round(total_at_10/(i+1), 5)}")


def flatten_dicts(batched_dict):
    flattened_dict = defaultdict(list)
    for batch in batched_dict:
        for key, val in batch.items():
            flattened_dict[key].extend(val)
    return flattened_dict


def log_setup(args):
    """Create log directory and logger. Log basic set up information."""
    # wait for first process to create log directory and dump config
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(device_ids=[args.local_rank])

    if args.local_rank in [-1, 0]:
        os.makedirs(args.log_dir, exist_ok=True)

        # dump args as config
        # https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
        with open(os.path.join(args.log_dir, "config.json"), "w") as f:
            ujson.dump(dict(sorted(args.__dict__.items())), f)

        if args.local_rank != -1:
            torch.distributed.barrier(device_ids=[args.local_rank])

    # write separate log for each process with local rank tags
    log_path = (
        os.path.join(args.log_dir, f"log_{args.local_rank}.txt")
        if args.local_rank != -1
        else os.path.join(args.log_dir, "log.txt")
    )
    if args.local_rank in [-1, 0]:
        # only use streamhandler for first process
        handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
    else:
        handlers = [
            logging.FileHandler(log_path),
        ]
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    # dump git hash
    label = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    logger.info(f"Git hash: {label}")

    # dump basic machine info
    machine_info = os.uname()
    logger.info(f"Machine: {machine_info.nodename} ({machine_info.version})")

    # dump run cmd
    cmd_msg = " ".join(sys.argv)
    logger.info(f"CMD: {cmd_msg}")
