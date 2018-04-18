from contextlib import contextmanager
from typing import List, Optional

from torch.autograd import Variable as Var
from torchcrf import CRF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class FeedforwardTagger(nn.Module):
    def __init__(self,
                 num_words: int,
                 num_tags: int,
                 word_embedding_size: int = 100,
                 window: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.5,
                 padding_idx: int = 0,
                 use_crf: bool = False,
                 ) -> None:
        super().__init__()

        self.num_words = num_words
        self.num_tags = num_tags
        self.word_embedding_size = word_embedding_size
        self.window = window
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.word_embedding = nn.Embedding(num_words, word_embedding_size, padding_idx=padding_idx)
        num_ctx_words = 2 * window + 1
        self.ff = nn.Sequential(
            nn.Linear(word_embedding_size * num_ctx_words, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )
        self.crf = CRF(num_tags) if use_crf else None
        self.reset_parameters()

    @property
    def use_crf(self) -> bool:
        return self.crf is not None

    def reset_parameters(self) -> None:
        init.xavier_uniform(self.ff[0].weight, gain=init.calculate_gain('tanh'))
        init.constant(self.ff[0].bias, 0)
        init.xavier_uniform(self.ff[-1].weight)
        init.constant(self.ff[-1].bias, 0)
        if self.use_crf:
            self.crf.reset_parameters()

    def forward(self, words: Var, tags: Var, mask: Optional[Var] = None) -> Var:
        self._check_dims_and_sizes(words, tags=tags, mask=mask)
        # shape: (batch_size, seq_length, num_tags)
        emissions = self._compute_emissions(words)
        if self.use_crf:
            # shape: (batch_size,)
            return self._compute_crf_loss(emissions, tags, mask=mask)
        # shape: (batch_size,)
        return self._compute_cross_entropy_loss(emissions, tags)

    def decode(self, words: Var, mask: Optional[Var] = None) -> List[List[int]]:
        self._check_dims_and_sizes(words, mask=mask)
        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self._compute_emissions(words)
            if self.use_crf:
                predictions = self._decode_with_crf(emissions, mask=mask)
            else:
                predictions = self._greedy_decode(emissions, mask=mask)
        return predictions

    def _check_dims_and_sizes(self,
                              words: Var,
                              tags: Optional[Var] = None,
                              mask: Optional[Var] = None,
                              ) -> None:
        if words.dim() != 2:
            raise ValueError('expected words to have dim of 2, got', words.dim())
        if tags is not None and words.size() != tags.size():
            raise ValueError(
                'expected words and tags to have the same size (batch_size, seq_length)')
        if mask is not None and mask.size() != words.size():
            raise ValueError(
                'expected words and mask to have the same size (batch_size, seq_length)')

    def _compute_emissions(self, words: Var) -> Var:
        assert words.dim() == 2

        batch_size, seq_length = words.size()

        padded = torch.cat((self._get_padding(batch_size), words), 1)  # pad left
        padded = torch.cat((padded, self._get_padding(batch_size)), 1)  # pad right
        # shape: (batch_size, seq_length + 2*window, word_embedding_size)
        embedded = self.word_embedding(padded)

        result = []
        for i in range(seq_length):
            lo, hi = i, i + 2 * self.window + 1
            inputs = embedded[:, lo:hi, :].contiguous()
            # shape: (batch_size, (2*window + 1) * word_embedding_size)
            inputs = inputs.view(batch_size, -1)
            # shape: (batch_size, num_tags)
            outputs = self.ff(inputs)
            result.append(outputs)
        # shape: (batch_size, seq_length, num_tags)
        return torch.stack(result, dim=1)

    def _compute_crf_loss(self, emissions: Var, tags: Var, mask: Optional[Var] = None) -> Var:
        assert emissions.dim() == 3
        assert emissions.size()[:-1] == tags.size()
        assert mask is None or tags.size() == mask.size()

        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()
        # shape: (seq_length, batch_size)
        tags = tags.transpose(0, 1).contiguous()
        if mask is not None:
            # shape: (seq_length, batch_size)
            mask = mask.transpose(0, 1).contiguous()
        # shape: (batch_size,)
        return -self.crf(emissions, tags, mask=mask, reduce=False)

    def _compute_cross_entropy_loss(self, emissions: Var, tags: Var) -> Var:
        assert emissions.dim() == 3
        assert emissions.size()[:-1] == tags.size()

        # shape: (batch_size * seq_length, num_tags)
        emissions = emissions.view(-1, emissions.size(-1))
        # shape: (batch_size * seq_length,)
        tags = tags.view(-1)
        # shape: (batch_size,)
        return F.cross_entropy(emissions, tags, ignore_index=self.padding_idx, reduce=False)

    def _decode_with_crf(self, emissions: Var, mask: Optional[Var] = None) -> List[List[int]]:
        assert emissions.dim() == 3
        assert mask is None or emissions.size()[:-1] == mask.size()

        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()
        if mask is not None:
            # shape: (seq_length, batch_size)
            mask = mask.transpose(0, 1).contiguous()
        return self.crf.decode(emissions, mask=mask)

    def _greedy_decode(self,
                       emissions: Var,
                       mask: Optional[Var] = None,
                       ) -> List[List[int]]:
        assert emissions.dim() == 3
        assert mask is None or emissions.size()[:-1] == mask.size()

        # shape: (batch_size, seq_length)
        _, best_tags = emissions.max(dim=-1)
        if mask is None:
            # shape: (batch_size, seq_length)
            mask = self._get_mask_for(best_tags)

        best_tags, mask = best_tags.data, mask.data
        result = []
        for best, mask_ in zip(best_tags, mask):
            length = mask_.long().sum()
            result.append(best[:length].tolist())
        return result

    def _get_padding(self, batch_size: int) -> Var:
        return Var(torch.Tensor(batch_size, self.window).fill_(self.padding_idx).long())

    def _get_mask_for(self, x: Var) -> Var:
        return Var(x.data.new(x.size()).fill_(1).byte())


@contextmanager
def evaluation(model):
    training = model.training
    model.eval()
    yield
    model.train(training)
