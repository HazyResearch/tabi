import json
import logging
import pickle
import unicodedata
from collections import Counter

import jsonlines
import numpy as np
import torch
import torch.nn.functional as F

from tabi.constants import MENTION_END, MENTION_START

logger = logging.getLogger(__name__)


def load_neg_samples(neg_sample_file, num_negatives, data_len):
    with open(neg_sample_file) as f:
        neg_samples = json.load(f)
        # store only negative samples we need in array
        assert len(neg_samples) == data_len, f"{data_len} {len(neg_samples)}"
        ns_array = np.zeros((len(neg_samples), num_negatives), dtype="int64")
        for ns in neg_samples:
            assert len(neg_samples[ns]["samples"]) >= num_negatives
            ns_array[int(ns)] = neg_samples[ns]["samples"][:num_negatives]
        return ns_array


def save_entity_map(id_map_path, entity_ids):
    """Save mapping of embedding index to entity id (eid)"""
    logger.info(f"Saving ids to {id_map_path}.")
    entity_map = {int(idx): int(eid) for idx, eid in enumerate(entity_ids)}
    with open(id_map_path, "wb") as f:
        pickle.dump(entity_map, f)


def convert_types_to_onehot(types, type_vocab):
    types = [type_vocab[t] for t in types]
    if len(types) == 0:
        one_hot_types = torch.zeros(len(type_vocab))
    else:
        one_hot_types = torch.sum(
            F.one_hot(torch.tensor(types), num_classes=len(type_vocab)), dim=0
        ).float()
    return one_hot_types


def load_entity_data(datapath: str):
    """Load entity data as dictionary with entity ids as keys"""
    data = {}
    if datapath.endswith(".pkl"):
        with open(datapath, "rb") as f:
            return pickle.load(f)
    with jsonlines.open(datapath) as f:
        for line_idx, line in enumerate(f):
            lid = line["label_id"] if "label_id" in line else line_idx
            data[lid] = {
                "id": lid,
                "description": line.get("text", ""),
                "title": line.get("title", ""),
                "types": line.get("types", []),
                "wikipedia_page_id": line.get("wikipedia_page_id", ""),
            }
    # save as pickle for faster loading
    entityfile = datapath.split(".jsonl")[0] + ".pkl"
    with open(entityfile, "wb") as f:
        pickle.dump(data, f)
        logger.debug(f"Wrote entities to {entityfile}")
    return data


def get_prepped_type_file(datapath: str):
    if datapath.endswith(".pkl"):
        tag = datapath.split(".pkl")[0]
    if datapath.endswith(".jsonl"):
        tag = datapath.split(".jsonl")[0]
    return f"{tag}_onehot_types.npy"


def load_data(datapath):
    samples = []
    ent_counter = Counter()
    with jsonlines.open(datapath) as f:
        for line in f:
            # each mention gets its own example
            if len(line["mentions"]) > 0:
                for i in range(len(["mentions"])):
                    sample = {
                        "id": line["id"],
                        "gold": line["label_id"][i],
                        "alt_gold": line["alt_label_id"][i]
                        if "alt_label_id" in line
                        else [],
                        "text": line["text"],
                        # keep track of all mentions that were present
                        "char_spans": line["mentions"][i],
                    }
                    ent_counter[sample["gold"]] += 1
                    samples.append(sample)
            else:
                sample = {
                    "id": line["id"],
                    "gold": line["label_id"][0],
                    "alt_gold": line["alt_label_id"][0]
                    if "alt_label_id" in line
                    else [],
                    "text": line["text"],
                    "char_spans": [],
                }
                ent_counter[sample["gold"]] += 1
                samples.append(sample)
    return samples, ent_counter


# modified from https://github.com/facebookresearch/BLINK/blob/main/blink/biencoder/data_process.py
def get_context_window(char_spans, tokenizer, context, max_context_length):
    start_idx = char_spans[0]
    end_idx = char_spans[1]
    context_left = context[:start_idx]
    mention = context[start_idx:end_idx]
    context_right = context[end_idx:]
    mention_tokens = [MENTION_START] + tokenizer.tokenize(mention) + [MENTION_END]
    context_left_tokens = tokenizer.tokenize(context_left)
    context_right_tokens = tokenizer.tokenize(context_right)
    left_quota = (max_context_length - len(mention_tokens)) // 2 - 1
    right_quota = max_context_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add
    context_tokens = (
        context_left_tokens[-left_quota:]
        + mention_tokens
        + context_right_tokens[:right_quota]
    )
    return context_tokens


def clean_spaces(context, char_spans):
    """Update char span to require mention to not start with a space"""
    while unicodedata.normalize("NFKD", context[char_spans[0]]) == " ":
        char_spans[0] += 1
    assert char_spans[1] > char_spans[0]
    return char_spans


def load_types(type_path):
    types = []
    with open(type_path) as f:
        for line in f:
            types.append(line.strip())
    return types
