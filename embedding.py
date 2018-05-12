from typing import Optional

from torch.autograd import Variable as Var
import torch
import torch.nn as nn
import torch.nn.init as init


class Embedding(nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int = 0,
            dropout: float = 0.5,
    ) -> None:
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Var) -> Var:
        return self.dropout(super(Embedding, self).forward(inputs))


class EmbeddingWithPretrained(Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int = 0,
            dropout: float = 0.5,
            pretrained_embedding: Optional[torch.Tensor] = None,
    ) -> None:
        self.pretrained_embedding = None
        super(EmbeddingWithPretrained, self).__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx, dropout=dropout)

        if pretrained_embedding is not None:
            n, sz = pretrained_embedding.size()
            self.pretrained_embedding = nn.Embedding(n, sz)
            self.pretrained_embedding.weight.data = pretrained_embedding
            self.pretrained_embedding.weight.requires_grad = False  # prevent update when training
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim + sz, embedding_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super(EmbeddingWithPretrained, self).reset_parameters()
        if self.pretrained_embedding is not None:
            init.xavier_uniform(
                self.embedding_projection[0].weight, gain=init.calculate_gain('tanh'))
            init.constant(self.embedding_projection[0].bias, 0.)

    def forward(self, inputs: Var) -> Var:
        # inputs shape: (batch_size, seq_length)

        # shape: (batch_size, seq_length, embedding_dim)
        embedded = super(EmbeddingWithPretrained, self).forward(inputs)

        if self.pretrained_embedding is not None:
            # shape: (batch_size, seq_length, sz)
            pretrained = self.pretrained_embedding(inputs)
            # shape: (batch_size, seq_length, embedding_dim + sz)
            embedded = torch.cat((embedded, pretrained), dim=-1)
            # shape: (batch_size, seq_length, embedding_dim)
            embedded = self.embedding_projection(embedded)

        return embedded
