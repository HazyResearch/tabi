import torch
import torch.nn as nn

import tabi.utils.train_utils as train_utils


class TABiLoss(nn.Module):
    """
    Type-Aware Bi-encoders (TABi) loss.
    """

    def __init__(self, temperature=0.01, alpha=0.1, type_equivalence="strict"):
        super(TABiLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.type_equivalence = type_equivalence

    def forward(
        self,
        query_embs,
        entity_embs,
        query_type_labels,
        query_entity_labels,
        entity_labels,
    ):
        """Computes TABi loss.

        Args:
            query_embs: (num_queries, hidden_dim)
            entity_embs: (num_uniq_entities, hidden_dim)
            query_type_labels: (num_queries, num_types)
            query_entity_labels: (num_queries)
            entity_labels: (num_uniq_entities)

        Returns:
            dict of {type_loss, ent_loss, total_loss}
        """
        type_loss = torch.tensor(0.0)
        ent_loss = torch.tensor(0.0)
        total_loss = torch.tensor(0.0)
        # entity loss
        if self.alpha < 1.0:
            all_embs = torch.cat([query_embs, entity_embs], dim=0)
            all_ent_labels = torch.cat(
                [query_entity_labels, entity_labels], dim=0
            ).view(-1, 1)
            # -1 means no label
            no_label_mask = all_ent_labels == -1
            same_label_mask = torch.eq(all_ent_labels, all_ent_labels.T).bool()
            ent_loss = SupConLoss(temperature=self.temperature)(
                embs=all_embs,
                same_label_mask=same_label_mask,
                no_label_mask=no_label_mask,
            )
        # type loss
        if self.alpha > 0.0:
            no_label_mask = query_type_labels.sum(1) == 0
            same_label_mask = train_utils.get_type_label_mask(
                query_type_labels, type_equivalence=self.type_equivalence
            ).bool()
            type_loss = SupConLoss(temperature=self.temperature)(
                embs=query_embs,
                no_label_mask=no_label_mask,
                same_label_mask=same_label_mask,
            )
        total_loss = type_loss * self.alpha + ent_loss * (1 - self.alpha)
        return {"total_loss": total_loss, "ent_loss": ent_loss, "type_loss": type_loss}


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Modified from https://github.com/HobbitLong/SupContrast.
    """

    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embs, same_label_mask, no_label_mask):
        """Compute supervised contrastive loss (variant).

        Args:
            embs: (num_examples, hidden_dim)
            same_label_mask: (num_examples, num_examples)
            no_label_mask: (num_examples)

        Returns:
            A loss scalar
        """
        # compute similarity scores for embs
        sim = embs @ embs.T / self.temperature
        # for numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # compute log-likelihood for each pair
        # ***unlike original supcon, do not include examples with the same label in the denominator***
        negs = torch.exp(sim) * ~(same_label_mask)
        denominator = negs.sum(axis=1, keepdim=True) + torch.exp(sim)
        # log(exp(x)) = x and log(x/y) = log(x) - log(y)
        log_prob = sim - torch.log(denominator)

        # compute mean of log-likelihood over all positive pairs for a query/entity
        # exclude self from positive pairs
        pos_pairs_mask = same_label_mask.fill_diagonal_(0)
        # only include examples in loss that have positive pairs and the class label is known
        include_in_loss = (pos_pairs_mask.sum(1) != 0) & (~no_label_mask).flatten()
        # we add ones to the denominator to avoid nans when there are no positive pairs
        mean_log_prob_pos = (pos_pairs_mask * log_prob).sum(1) / (
            pos_pairs_mask.sum(1) + (~include_in_loss).float()
        )
        mean_log_prob_pos = mean_log_prob_pos[include_in_loss]

        # return zero loss if there are no values to take average over
        if mean_log_prob_pos.shape[0] == 0:
            return torch.tensor(0)

        # scale loss by temperature (as done in supcon paper)
        loss = -1 * self.temperature * mean_log_prob_pos

        # average loss over all queries/entities that have at least one positive pair
        return loss.mean()
