import json
import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import logging as tf_logging

import tabi.utils.data_utils as data_utils
import tabi.utils.utils as utils
from tabi.constants import ENT_START, MENTION_END, MENTION_START

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings("ignore", message=".*The given NumPy array is not writeable.*")

# suppress warnings from huggingface
tf_logging.set_verbosity_error()


class EntityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        entity_path: Optional[str] = None,
        max_entity_length: int = 128,
        add_entity_type_in_description: bool = False,
        tokenized_entity_data: Optional[str] = None,
        tokenizer_name: str = "bert-base-uncased",
        type_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # hyperparameters
        self.max_entity_length = max_entity_length
        self.add_entity_type_in_description = add_entity_type_in_description

        # load entity file
        logger.debug("Loading entity data...")
        self.entity_cache = data_utils.load_entity_data(entity_path)
        self.entity_ids = np.array(list(self.entity_cache.keys()))
        self.entity_ids.flags.writeable = False

        # get typename to id map to pass types by id in batch
        # type ids are used for masking in loss function
        if type_path is not None:
            self.all_types = np.array(data_utils.load_types(type_path))
            self.all_types.flags.writeable = False
            self.type_vocab = {name: i for i, name in enumerate(self.all_types)}

        self.eid2row = {eid: row_id for row_id, eid in enumerate(self.entity_ids)}
        logger.debug("Finished loading entity data!")

        # load tokenized entity data if available
        self.tokenized_entity_data = None
        if tokenized_entity_data is not None:
            logger.info(
                f"Loading preprocessed entity data from {tokenized_entity_data}."
            )

            # load read only memory mapped file of tokenized entity data
            self.tokenized_entity_data = np.memmap(
                tokenized_entity_data,
                mode="r",
                shape=len(self.entity_ids),
                dtype=utils.get_mmap_type(max_entity_length),
            )

            # map all types to a read only numpy array
            if not os.path.exists(data_utils.get_prepped_type_file(entity_path)):
                self.all_one_hot_types = np.zeros(
                    (len(self.entity_ids), len(self.all_types))
                )
                for e in self.entity_cache:
                    ent_typeids = [
                        self.type_vocab[t] for t in self.entity_cache[e]["types"]
                    ]
                    if len(ent_typeids) > 0:
                        row_id = self.eid2row[e]
                        self.all_one_hot_types[row_id] = torch.sum(
                            F.one_hot(
                                torch.tensor(ent_typeids),
                                num_classes=len(self.all_types),
                            ),
                            dim=0,
                        )
                type_mmap = np.memmap(
                    data_utils.get_prepped_type_file(entity_path),
                    mode="w+",
                    shape=self.all_one_hot_types.shape,
                )
                type_mmap[:] = self.all_one_hot_types
            else:
                self.all_one_hot_types = np.memmap(
                    data_utils.get_prepped_type_file(entity_path), mode="r"
                ).reshape(len(self.entity_ids), len(self.all_types))
            self.all_one_hot_types.flags.writeable = False

            # no longer need entity cache
            self.entity_cache = None

        # set up tokenizer
        logger.info(f"Using tokenizer: {tokenizer_name}")

        # use local files unless not present
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, local_files_only=True
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENT_START, MENTION_START, MENTION_END]}
        )

    def __len__(self) -> int:
        return len(self.entity_ids)

    def __getitem__(self, index):
        entity_id = self.entity_ids[index]
        # get entity tokens from preprocessed entity data
        if self.tokenized_entity_data is not None:
            entity_tokens = self.tokenized_entity_data[index]
            # convert memory mapped format to standard dict format
            entity_tokens = {
                key: entity_tokens[key] for key in entity_tokens.dtype.names
            }
        # tokenize entity data on the fly
        else:
            entity_data = self.entity_cache[entity_id]
            entity_tokens = self.get_entity_tokens(entity_data)
            entity_tokens = {k: v[0] for k, v in entity_tokens.items()}
        return {"sample": entity_tokens, "entity_id": entity_id}

    def get_entity_tokens(self, entity):
        title = entity["title"]
        entity_description = entity["description"]
        if self.add_entity_type_in_description:
            type_str = utils.get_type_str(entity["types"])
            ent_str = (
                title
                + " "
                + ENT_START
                + " "
                + type_str
                + " "
                + ENT_START
                + " "
                + entity_description
            )
        else:
            ent_str = title + " " + ENT_START + " " + entity_description
        inputs = self.tokenizer(
            ent_str,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_entity_length,
        )
        return inputs

    def get_types(self, ent):
        if self.tokenized_entity_data is None:
            one_hot_types = data_utils.convert_types_to_onehot(
                types=self.entity_cache[ent]["types"], type_vocab=self.type_vocab
            )
        else:
            one_hot_types = self.all_one_hot_types[self.eid2row[ent]]
        return np.array(one_hot_types)


class EntityLinkingDataset(EntityDataset):
    def __init__(
        self,
        max_entity_length: int = 128,
        max_context_length: int = 64,
        data_path: str = "",
        entity_path: str = "",
        neg_sample_file: Optional[str] = None,
        num_negatives: int = 0,
        add_entity_type_in_description: bool = False,
        top_k: int = 10,
        tokenize_entities: bool = True,
        tokenized_entity_data: Optional[str] = None,
        tokenizer_name: str = "bert-base-uncased",
        type_path: Optional[str] = None,
        is_eval: bool = False,
    ) -> None:
        super().__init__(
            max_entity_length=max_entity_length,
            entity_path=entity_path,
            add_entity_type_in_description=add_entity_type_in_description,
            tokenized_entity_data=tokenized_entity_data,
            tokenizer_name=tokenizer_name,
            type_path=type_path,
        )

        # hyperparameters
        self.eval = is_eval
        self.max_context_length = max_context_length
        self.top_k = top_k
        self.tokenize_entities = tokenize_entities

        # load context files
        logger.debug("Loading context data...")
        self.data, self.ent_counter = data_utils.load_data(data_path)
        logger.debug("Finished loading context data!")

        # load hard negative samples
        self.neg_sample_file = neg_sample_file
        self.num_negatives = num_negatives
        self.neg_samples = None
        if self.num_negatives > 0 and self.neg_sample_file is not None:
            logger.info(
                f"Using {self.num_negatives} hard negatives from {self.neg_sample_file}"
            )
            self.neg_samples = data_utils.load_neg_samples(
                self.neg_sample_file,
                data_len=len(self.data),
                num_negatives=self.num_negatives,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # get mention context data
        context_tokens = defaultdict(list)
        tokens = self.get_context_tokens(
            sample["text"], char_spans=sample["char_spans"]
        )
        for key in tokens.keys():
            context_tokens[key].append(tokens[key][0])
        context_tokens = {k: torch.stack(v) for k, v in context_tokens.items()}

        # get entity data
        gold = sample["gold"]
        entities = [gold]

        # allow up to 10 alternate gold
        alt_gold = sample["alt_gold"][:10]
        # pad alt gold with gold so all the same size for batching
        while len(alt_gold) < 10:
            alt_gold.append(gold)

        if self.neg_samples is not None:
            for ent in self.neg_samples[index]:
                entities.append(ent)

        # get entity tokens
        entity_tokens = defaultdict(list)
        if self.tokenize_entities:
            # tokenize entity data on the fly
            if self.tokenized_entity_data is None:
                for entity in entities:
                    tokens = self.get_entity_tokens(self.entity_cache[entity])
                    for key in tokens.keys():
                        entity_tokens[key].append(tokens[key][0])
            # use preprocessed entity data
            else:
                for entity in entities:
                    tokens = self.tokenized_entity_data[self.eid2row[entity]]
                    for key in tokens.dtype.names:
                        # this throws a warning about torch tensors not being read-only
                        # we do not copy as tokenized_entity_data is large
                        entity_tokens[key].append(torch.from_numpy(tokens[key]))
            entity_tokens = {k: torch.stack(v) for k, v in entity_tokens.items()}

        # use query type labels if training
        query_type_labels = (
            torch.from_numpy(self.get_types(sample["gold"])) if not self.eval else []
        )

        return {
            "data_id": index,
            "context": context_tokens,
            "entity": entity_tokens,
            "entity_labels": torch.tensor(entities),
            "query_entity_labels": torch.tensor(gold),
            "query_type_labels": query_type_labels,
            "alt_gold": torch.tensor(alt_gold),
        }

    def get_context_tokens(self, context, char_spans=[]):
        # no mention boundaries
        if len(char_spans) == 0:
            return self.tokenizer(
                context,
                padding="max_length",
                add_special_tokens=True,
                return_tensors="pt",  # return as pytorch tensors
                truncation=True,
                max_length=self.max_context_length,
            )
        char_spans = data_utils.clean_spaces(context=context, char_spans=char_spans)
        context_tokens = data_utils.get_context_window(
            char_spans=char_spans,
            tokenizer=self.tokenizer,
            context=context,
            max_context_length=self.max_context_length,
        )
        # convert back to string to use tokenizer to pad and generate attention mask
        context = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(context_tokens)
        )
        return self.tokenizer(
            context,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",  # return as pytorch tensors
            truncation=True,
            max_length=self.max_context_length,
        )
