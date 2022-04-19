import logging

import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

logger = logging.getLogger()


class Encoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        logger.info(f"Using encoder model: {model_name}")
        # use local model files unless not present
        try:
            self.transformer = AutoModel.from_pretrained(
                model_name, local_files_only=True
            )
        except:
            self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.embeddings.word_embeddings.weight.size(1)

    def forward(self, x):
        input_ids = x["input_ids"]
        token_type_ids = x["token_type_ids"]
        attention_mask = x["attention_mask"]
        seq_len = input_ids.shape[-1]
        last_hidden_state, _ = self.transformer(
            input_ids=input_ids.reshape(-1, seq_len),
            token_type_ids=token_type_ids.reshape(-1, seq_len),
            attention_mask=attention_mask.reshape(-1, seq_len),
            return_dict=False,
        )
        return last_hidden_state


class Aggregator(nn.Module):
    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        logger.debug(f"L2 normalization: {normalize}")

    def forward(self, last_hidden_state):
        # take CLS token as embedding
        emb = last_hidden_state[:, 0]
        if self.normalize:
            return F.normalize(emb, p=2, dim=-1)
        return emb
