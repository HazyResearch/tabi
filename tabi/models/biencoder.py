import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tabi.models.layers import Aggregator, Encoder
from tabi.models.losses import TABiLoss
from tabi.utils.train_utils import first_nonzero, gather_embs

logger = logging.getLogger(__name__)


class Biencoder(nn.Module):
    def __init__(
        self,
        tied,
        temperature=0.1,
        top_k=10,
        entity_emb_path=None,
        model_name="bert-base-uncased",
        normalize=True,
        alpha=0.1,
        is_distributed=False,
    ):

        super(Biencoder, self).__init__()

        # initialize encoders
        if tied:
            # use same weights for entity encoder and query encoder
            self.entity_encoder = Encoder(model_name=model_name)
            self.query_encoder = self.entity_encoder
        else:
            self.entity_encoder = Encoder(model_name=model_name)
            self.query_encoder = Encoder(model_name=model_name)

        self.dim = self.entity_encoder.output_dim

        self.entity_aggregator = Aggregator(normalize=normalize)
        self.query_aggregator = Aggregator(normalize=normalize)

        self.entity_embs = None
        if entity_emb_path is not None:
            entity_embs = np.memmap(entity_emb_path, dtype="float32", mode="r").reshape(
                -1, self.dim
            )
            self.entity_embs = torch.from_numpy(np.copy(entity_embs))

        # hyperparameters
        self.temperature = temperature
        self.top_k = top_k
        self.alpha = alpha
        self.is_distributed = is_distributed
        logger.debug(f"Using distributed for model: {is_distributed}")

        self.tabi_loss = TABiLoss(temperature=temperature, alpha=self.alpha)

    def _embed_entity(self, entity_data):
        """Get entity embeddings"""
        return self.entity_aggregator(self.entity_encoder(entity_data))

    def _embed_query(self, context_data):
        """Get query embeddings"""
        return self.query_aggregator(self.query_encoder(context_data))

    def forward(self, entity_data=None, context_data=None):
        entity_embs = None
        query_embs = None

        if entity_data is not None:
            entity_embs = self._embed_entity(entity_data)

        if context_data is not None:
            query_embs = self._embed_query(context_data)

        return {"entity_embs": entity_embs, "query_embs": query_embs}

    def loss(
        self,
        query_embs,
        entity_embs,
        query_entity_labels,
        entity_labels,
        query_type_labels=None,
    ):
        """
        Args:
            query_embs: (num_queries, hidden_dim)
            entity_embs: (num_entities, hidden_dim)
            query_entity_labels: (num_queries)
            entity_labels: (num_queries, num_negatives+1)
            query_type_labels: (num_queries, num_types)

        num_entities will include duplicate entities if the same entity
        occurs with more than one example in the batch.

        Returns:
            dict of {type_loss, ent_loss, total_loss}
        """
        # get embs across gpus before computing loss if distributed
        if self.is_distributed:
            entity_embs = gather_embs(entity_embs)
            query_embs = gather_embs(query_embs)
            query_type_labels = gather_embs(query_type_labels)
            query_entity_labels = gather_embs(query_entity_labels)
            entity_labels = gather_embs(entity_labels)

        # flatten entity labels to have same dimension 0 as entity embs
        entity_labels = entity_labels.flatten()

        # remove duplicate entities from a batch
        uniq_ent_indices = torch.unique(
            first_nonzero(
                torch.eq(
                    entity_labels.unsqueeze(1), entity_labels.unsqueeze(1).T
                ).float(),
                axis=1,
            )[1]
        )
        entity_embs = entity_embs[uniq_ent_indices]
        entity_labels = entity_labels[uniq_ent_indices]

        return self.tabi_loss(
            query_embs=query_embs,
            entity_embs=entity_embs,
            query_type_labels=query_type_labels,
            query_entity_labels=query_entity_labels,
            entity_labels=entity_labels,
        )

    def predict(self, context_data, data_id=None):
        query_embs = self(context_data=context_data)["query_embs"]
        # get top entity candidates using nearest neighbor search
        scores = query_embs @ self.entity_embs.t()
        query_entity_scores, index = scores.topk(self.top_k)
        probs = F.softmax(query_entity_scores / self.temperature, -1)
        # "indices" are row in the embedding matrix and may not correspond to entity_ids
        # we convert back to entity_ids in saving predictions/negative samples (see utils.py)
        prediction = {
            "scores": query_entity_scores,
            "probs": probs,
            "indices": index,
            "data_id": data_id,
        }
        prediction = {
            key: value.cpu().numpy()
            for key, value in prediction.items()
            if len(value) > 0
        }
        return prediction
