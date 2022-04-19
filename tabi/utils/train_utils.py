import logging
import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.optimization import AdamW

logger = logging.getLogger(__name__)


def get_type_label_mask(labels, type_equivalence="strict"):
    """Generate the mask indicating which queries have the same type.

    Args:
        labels: (num_queries, num_types)
        type_equivalence (str): 'strict' or 'weak'

    Returns:
        mask with 1 where two queries share the same type and 0 otherwise
    """
    # weak equivalence
    # two sets of types are considered equivalent if more than
    # 50% of the types are shared between them (based on cardinality of larger set)
    if type_equivalence == "weak":
        shared_labels = torch.matmul(labels.float(), labels.float().T)
        num_types_per_el = labels.sum(1)
        max_types = (
            torch.cartesian_prod(num_types_per_el, num_types_per_el)
            .max(1)[0]
            .reshape(num_types_per_el.shape[0], -1)
        )
        same_label_mask = (shared_labels > max_types * 0.5).float()
    # strict equivalence
    # two sets of types are considered equivalent if all types match
    else:
        shared_labels = torch.matmul(labels.float(), labels.float().T)
        labels_cols = labels.sum(1).repeat(len(labels), 1)
        labels_rows = labels.sum(1).unsqueeze(1).repeat(1, len(labels))
        same_label_mask = (
            torch.eq(shared_labels, labels_rows) & torch.eq(shared_labels, labels_cols)
        ).float()
    return same_label_mask


# https://discuss.pytorch.org/t/first-nonzero-index/24769
def first_nonzero(x, axis=0):
    nonz = x > 0
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)


# modified from https://github.com/facebookresearch/BLINK/blob/main/blink/common/optimizer.py
def get_bert_optimizer(model, learning_rate):
    """Optimizes the network with AdamWithDecay"""
    parameters_with_decay = []
    parameters_without_decay = []
    for param_name, param in model.named_parameters():
        # do not use decay on bias terms
        if "bias" in param_name:
            parameters_without_decay.append(param)
        else:
            parameters_with_decay.append(param)
    optimizer_grouped_parameters = [
        {"params": parameters_with_decay, "weight_decay": 0.01},
        {"params": parameters_without_decay, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        correct_bias=False,
        no_deprecation_warning=True,
    )
    return optimizer


def gather_embs(embs):
    """
    Gathers embeddings across machines in distributed training
    and combines into a single embedding.
    """
    return torch.cat(GatherLayer.apply(embs.contiguous()), dim=0)


# https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def set_random_seed(random_seed=0):
    logger.info(f"Random seed: {random_seed}")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
