from collections import Counter
from contextlib import contextmanager
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple

from torch.autograd import Variable as Var
from torchcrf import CRF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


Word = str
Tag = str


class MemorizationTagger(object):
    def __init__(self, mapping: Mapping[FrozenSet, Tag], window: int = 2) -> None:
        if not mapping:
            raise ValueError('mapping cannot be empty')

        self.mapping = mapping
        self.window = window

        c = Counter(self.mapping.values())
        assert c
        self._most_common_value = c.most_common(n=1)[0][0]

    @classmethod
    def train(cls,
              tagged_sents: Sequence[Sequence[Tuple[Word, Tag]]],
              window: int = 2,
              ) -> 'MemorizationTagger':
        mapping = {}
        for tagged_sent in tagged_sents:
            words, tags = zip(*tagged_sent)
            for fs, tag in zip(cls._extract_features(words, window=window), tags):
                mapping[frozenset(fs.items())] = tag
        return cls(mapping, window=window)

    def predict(self, sent: Sequence[Word]) -> Sequence[Tag]:
        prediction = []
        for fs in self._extract_features(sent, window=self.window):
            prediction.append(self._getitem(frozenset(fs.items())))
        return prediction

    def _getitem(self, key: FrozenSet) -> Tag:
        try:
            return self.mapping[key]
        except KeyError:
            return self._most_common_value

    @staticmethod
    def _extract_features(sent, window=2):
        for i in range(len(sent)):
            fs = {'w[0]': sent[i]}
            for d in range(1, window + 1):
                fs[f'w[-{d}]'] = sent[i - d] if i - d >= 0 else '<s>'
                fs[f'w[+{d}]'] = sent[i + d] if i + d < len(sent) else '</s>'
            yield fs


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
                 pretrained_embedding: Optional[torch.Tensor] = None,
                 ) -> None:
        super().__init__()

        self.word_embedding = nn.Embedding(num_words, word_embedding_size, padding_idx=padding_idx)
        num_ctx_words = 2 * window + 1
        self.ff = nn.Sequential(
            nn.Linear(word_embedding_size * num_ctx_words, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )
        self.crf = CRF(num_tags) if use_crf else None

        self.pretrained_embedding = None
        if pretrained_embedding is not None:
            num_pretrained_words, pretrained_embedding_size = pretrained_embedding.size()
            self.pretrained_embedding = nn.Embedding(
                num_pretrained_words, pretrained_embedding_size)
            self.pretrained_embedding.weight.data = pretrained_embedding
            self.pretrained_embedding.weight.requires_grad = False  # prevent update when training
            self.embedding_projection = nn.Sequential(
                nn.Linear(word_embedding_size + pretrained_embedding_size, word_embedding_size),
                nn.Tanh(),
                nn.Dropout(dropout),
            )

        self.reset_parameters()

    @property
    def word_embedding_size(self) -> int:
        return self.word_embedding.embedding_dim

    @property
    def window(self) -> int:
        assert isinstance(self.ff[0], nn.Linear)
        assert self.ff[0].in_features % self.word_embedding_size == 0
        return (self.ff[0].in_features // self.word_embedding_size - 1) // 2

    @property
    def padding_idx(self) -> int:
        return self.word_embedding.padding_idx

    @property
    def uses_crf(self) -> bool:
        return self.crf is not None

    @property
    def uses_pretrained_embedding(self) -> bool:
        return self.pretrained_embedding is not None

    def reset_parameters(self) -> None:
        init.xavier_uniform(self.ff[0].weight, gain=init.calculate_gain('tanh'))
        init.constant(self.ff[0].bias, 0)
        init.xavier_uniform(self.ff[-1].weight)
        init.constant(self.ff[-1].bias, 0)
        if self.uses_crf:
            self.crf.reset_parameters()
        if self.uses_pretrained_embedding:
            init.xavier_uniform(
                self.embedding_projection[0].weight, gain=init.calculate_gain('tanh'))
            init.constant(self.embedding_projection[0].bias, 0)

    def forward(self, words: Var, tags: Var, mask: Optional[Var] = None) -> Var:
        self._check_dims_and_sizes(words, tags=tags, mask=mask)
        # shape: (batch_size, seq_length, num_tags)
        emissions = self._compute_emissions(words)
        if self.uses_crf:
            # shape: (batch_size,)
            return self._compute_crf_loss(emissions, tags, mask=mask)
        # shape: (batch_size,)
        return self._compute_cross_entropy_loss(emissions, tags)

    def decode(self, words: Var, mask: Optional[Var] = None) -> List[List[int]]:
        self._check_dims_and_sizes(words, mask=mask)
        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self._compute_emissions(words)
            if self.uses_crf:
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

        # shape: (batch_size, seq_length + window)
        padded = torch.cat((self._get_padding(batch_size), words), 1)  # pad left
        # shape: (batch_size, seq_length + 2*window)
        padded = torch.cat((padded, self._get_padding(batch_size)), 1)  # pad right
        # shape: (batch_size, seq_length + 2*window, word_embedding_size)
        embedded = self._embed_words(padded)

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

    def _embed_words(self, words: Var) -> Var:
        assert words.dim() == 2
        # words shape: (batch_size, seq_length)

        # shape: (batch_size, seq_length, word_embedding_size)
        embedded = self.word_embedding(words)
        if self.uses_pretrained_embedding:
            assert self.pretrained_embedding is not None
            # shape: (batch_size, seq_length, pretrained_embedding_size)
            pretrained = self.pretrained_embedding(words)
            # shape: (batch_size, seq_length, word_embedding_size + pretrained_embedding_size)
            embedded = torch.cat((embedded, pretrained), dim=-1)
            # shape: (batch_size, seq_length, word_embedding_size)
            embedded = self.embedding_projection(embedded)
        return embedded

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
        p = next(self.parameters())
        return Var(p.data.new(batch_size, self.window).fill_(self.padding_idx).long())

    def _get_mask_for(self, x: Var) -> Var:
        return Var(x.data.new(x.size()).fill_(1).byte())


@contextmanager
def evaluation(model):
    training = model.training
    model.eval()
    yield
    model.train(training)
