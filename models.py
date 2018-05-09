from collections import Counter
from contextlib import contextmanager
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
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
                 num_prefixes: Optional[Tuple[int, int]] = None,
                 num_suffixes: Optional[Tuple[int, int]] = None,
                 word_embedding_size: int = 100,
                 prefix_embedding_size: Union[Tuple[int, int], int] = 20,
                 suffix_embedding_size: Union[Tuple[int, int], int] = 20,
                 window: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.5,
                 padding_idx: int = 0,
                 use_lstm: bool = False,
                 use_crf: bool = False,
                 pretrained_embedding: Optional[torch.Tensor] = None,
                 ) -> None:
        super(FeedforwardTagger, self).__init__()

        self.word_embedding = nn.Embedding(num_words, word_embedding_size, padding_idx=padding_idx)
        self.prefix_embedding = None
        self.suffix_embedding = None
        total_features_size = word_embedding_size
        total_features_size += word_embedding_size * window * 2

        self.dropout = nn.Dropout(dropout)

        if num_prefixes is not None:
            if isinstance(prefix_embedding_size, int):
                prefix_embedding_size = (prefix_embedding_size, prefix_embedding_size)
            self.prefix_embedding = nn.ModuleList([
                nn.Embedding(num_prefixes[0], prefix_embedding_size[0], padding_idx=padding_idx),
                nn.Embedding(num_prefixes[1], prefix_embedding_size[1], padding_idx=padding_idx),
            ])
            total_features_size += sum(prefix_embedding_size)

        if num_suffixes is not None:
            if isinstance(suffix_embedding_size, int):
                suffix_embedding_size = (suffix_embedding_size, suffix_embedding_size)
            self.suffix_embedding = nn.ModuleList([
                nn.Embedding(num_suffixes[0], suffix_embedding_size[0], padding_idx=padding_idx),
                nn.Embedding(num_suffixes[1], suffix_embedding_size[1], padding_idx=padding_idx),
            ])
            total_features_size += sum(suffix_embedding_size)

        ff_input_size = total_features_size

        self.lstm = None
        if use_lstm:
            self.lstm = nn.LSTM(total_features_size, hidden_size,
                                num_layers=2, bidirectional=True, batch_first=True)
            ff_input_size = 2 * hidden_size

        self.ff = nn.Sequential(
            nn.Linear(ff_input_size, hidden_size),
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
    def prefix_embedding_size(self) -> Optional[Tuple[int, int]]:
        if not self.uses_prefix:
            return None
        assert self.prefix_embedding is not None
        return (
            self.prefix_embedding[0].embedding_dim,
            self.prefix_embedding[1].embedding_dim,
        )

    @property
    def suffix_embedding_size(self) -> Optional[Tuple[int, int]]:
        if not self.uses_suffix:
            return None
        assert self.suffix_embedding is not None
        return (
            self.suffix_embedding[0].embedding_dim,
            self.suffix_embedding[1].embedding_dim,
        )

    @property
    def window(self) -> int:
        assert isinstance(self.ff[0], nn.Linear)
        total_features_size = self.lstm.input_size if self.uses_lstm else self.ff[0].in_features
        total_features_size -= sum(self.prefix_embedding_size) if self.uses_prefix else 0  # noqa: T484
        total_features_size -= sum(self.suffix_embedding_size) if self.uses_suffix else 0  # noqa: T484
        assert total_features_size % self.word_embedding_size == 0
        return (total_features_size // self.word_embedding_size - 1) // 2

    @property
    def padding_idx(self) -> int:
        return self.word_embedding.padding_idx

    @property
    def uses_prefix(self) -> bool:
        return self.prefix_embedding is not None

    @property
    def uses_suffix(self) -> bool:
        return self.suffix_embedding is not None

    @property
    def uses_lstm(self) -> bool:
        return self.lstm is not None

    @property
    def uses_crf(self) -> bool:
        return self.crf is not None

    @property
    def uses_pretrained_embedding(self) -> bool:
        return self.pretrained_embedding is not None

    def reset_parameters(self) -> None:
        self.word_embedding.reset_parameters()

        if self.uses_prefix:
            assert self.prefix_embedding is not None
            for emb in self.prefix_embedding:
                emb.reset_parameters()

        if self.uses_suffix:
            assert self.suffix_embedding is not None
            for emb in self.suffix_embedding:
                emb.reset_parameters()

        if self.uses_lstm:
            self.lstm.reset_parameters()

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

    def forward(self,
                words: Var,
                tags: Var,
                prefixes: Optional[Var] = None,
                suffixes: Optional[Var] = None,
                mask: Optional[Var] = None,
                ) -> Var:
        self._check_dims_and_sizes(
            words, prefixes=prefixes, suffixes=suffixes, tags=tags, mask=mask)

        # shape: (batch_size, seq_length, num_tags)
        emissions = self._compute_emissions(words, mask=mask, prefixes=prefixes, suffixes=suffixes)

        if self.uses_crf:
            # shape: (batch_size,)
            return self._compute_crf_loss(emissions, tags, mask=mask)

        if mask is None:
            # shape (batch_size, seq_length)
            mask = self._get_mask_for(words)
        # shape: (batch_size,)
        return self._compute_cross_entropy_loss(emissions, tags, mask)

    def decode(self,
               words: Var,
               prefixes: Optional[Var] = None,
               suffixes: Optional[Var] = None,
               mask: Optional[Var] = None,
               ) -> List[List[int]]:
        self._check_dims_and_sizes(words, prefixes=prefixes, suffixes=suffixes, mask=mask)
        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self._compute_emissions(words, mask=mask, prefixes=prefixes, suffixes=suffixes)
            if self.uses_crf:
                predictions = self._decode_with_crf(emissions, mask=mask)
            else:
                predictions = self._greedy_decode(emissions, mask=mask)
        return predictions

    def _check_dims_and_sizes(self,
                              words: Var,
                              prefixes: Optional[Var] = None,
                              suffixes: Optional[Var] = None,
                              tags: Optional[Var] = None,
                              mask: Optional[Var] = None,
                              ) -> None:
        if words.dim() != 2:
            raise ValueError('expected words to have dim of 2, got', words.dim())
        if self.uses_prefix and prefixes is None:
            raise ValueError('model uses prefix but prefixes are not passed')
        if self.uses_suffix and suffixes is None:
            raise ValueError('model uses suffix but suffixes are not passed')

        batch_size, seq_length = words.size()

        if prefixes is not None and prefixes.size() != (batch_size, seq_length, 2):
            raise ValueError(
                f'expected prefixes to have size of ({batch_size}, {seq_length}, 2), got',
                tuple(prefixes.size()))
        if suffixes is not None and suffixes.size() != (batch_size, seq_length, 2):
            raise ValueError(
                f'expected suffixes to have size of ({batch_size}, {seq_length}, 2), got',
                tuple(suffixes.size()))
        if tags is not None and words.size() != tags.size():
            raise ValueError(
                f'expected tags to have size of ({batch_size}, {seq_length}), got', tuple(tags.size()))
        if mask is not None and words.size() != mask.size():
            raise ValueError(
                f'expected mask to have size of ({batch_size}, {seq_length}), got', tuple(mask.size()))

    def _compute_emissions(self,
                           words: Var,
                           mask: Optional[Var] = None,
                           prefixes: Optional[Var] = None,
                           suffixes: Optional[Var] = None,
                           ) -> Var:
        assert words.dim() == 2
        batch_size, seq_length = words.size()
        assert mask is None or mask.size() == (batch_size, seq_length)
        assert prefixes is None or prefixes.size() == (batch_size, seq_length, 2)
        assert suffixes is None or suffixes.size() == (batch_size, seq_length, 2)

        if self.window > 0:
            # shape: (batch_size, seq_length + window)
            words = torch.cat((self._get_padding(batch_size), words), 1)  # pad left
            # shape: (batch_size, seq_length + 2*window)
            words = torch.cat((words, self._get_padding(batch_size)), 1)  # pad right
        
        # shape: (batch_size, seq_length + 2*window, word_embedding_size)
        embedded_words = self._embed_words(words)
        # shape: (batch_size, seq_length, total_prefix_embedding_size)
        embedded_prefixes = self._embed_prefixes(prefixes)
        # shape: (batch_size, seq_length, total_suffix_embedding_size)
        embedded_suffixes = self._embed_suffixes(suffixes)

        embedded_words = self.dropout(embedded_words)
        embedded_prefixes = self.dropout(embedded_prefixes)
        embedded_suffixes = self.dropout(embedded_suffixes)

        inputs = embedded_words
        if self.window > 0:
            temp = []
            for i in range(seq_length):
                lo, hi = i, i + 2 * self.window + 1
                # shape: (batch_size, (2*window + 1) * word_embedding_size)
                word_feats = embedded_words[:, lo:hi, :].contiguous().view(batch_size, -1)
                temp.append(word_feats)
            # shape: (batch_size, seq_length, (2*window + 1) * word_embedding_size)
            inputs = torch.stack(temp, dim=1)

        affix_features = None
        if embedded_prefixes is not None and embedded_suffixes is not None:
            # shape: (batch_size, seq_length, total_prefix_emb_size + total_suffix_emb_size)
            affix_features = torch.cat((embedded_prefixes, embedded_suffixes), dim=-1).contiguous()
        elif embedded_prefixes is not None:
            # shape: (batch_size, seq_length, total_prefix_emb_size)
            affix_features = embedded_prefixes
        elif embedded_suffixes is not None:
            # shape: (batch_size, seq_length, total_suffix_emb_size)
            affix_features = embedded_suffixes

        if affix_features is not None:
            # shape: (batch_size, seq_length, (2*window+1)*word_embedding_size+total_affix_emb_size)
            inputs = torch.cat((inputs, affix_features), dim=-1)

        # apply Bidirectional LSTM
        if self.uses_lstm:
            if mask is None:
                # shape (batch_size, seq_length)
                mask = self._get_mask_for(words)   
            
            seq_lengths = torch.sum(mask.int(), dim=1)
            seq_lengths, sent_perm = seq_lengths.sort(0, descending=True)
            inputs = inputs[sent_perm]

            packed_input = pack_padded_sequence(inputs, seq_lengths.data.tolist(), batch_first=True)

            lstm_out, _ = self.lstm(packed_input)      
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            seq_lengths, original_perm = sent_perm.sort(0, descending=False)
            inputs = lstm_out[original_perm]

        # shape: (batch_size, seq_length, num_tags)
        return self.ff(inputs)

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

    def _embed_prefixes(self, prefixes: Optional[Var] = None) -> Optional[Var]:
        return self._embed_affixes('prefix', affixes=prefixes)

    def _embed_suffixes(self, suffixes: Optional[Var] = None) -> Optional[Var]:
        return self._embed_affixes('suffix', affixes=suffixes)

    def _embed_affixes(self, name: str, affixes: Optional[Var] = None) -> Optional[Var]:
        assert not getattr(self, f'uses_{name}') or affixes is not None
        assert affixes is None or (affixes.dim() == 3 and affixes.size(-1) == 2)
        # affixes shape: (batch_size, seq_length, 2)
        if affixes is None:
            return None
        if not getattr(self, f'uses_{name}'):
            warnings.warn(f'the model does not use {name} features but {name} is passed, ignoring')
            return None

        assert getattr(self, f'{name}_embedding') is not None
        embedded = []
        for i in range(affixes.size(-1)):
            embedding = getattr(self, f'{name}_embedding')
            # shape: (batch_size, seq_length, affix_embedding_size[i])
            x = embedding[i](affixes[:, :, i])
            embedded.append(x)
        return torch.cat(embedded, dim=-1)

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

    def _compute_cross_entropy_loss(self, emissions: Var, tags: Var, mask: Var) -> Var:
        assert emissions.dim() == 3
        assert emissions.size()[:-1] == tags.size()
        assert mask.size() == tags.size()

        batch_size = emissions.size(0)

        # shape: (batch_size * seq_length, num_tags)
        emissions = emissions.view(-1, emissions.size(-1))
        # shape: (batch_size * seq_length,)
        tags = tags.view(-1)
        # shape: (batch_size * seq_length,)
        entropy = F.cross_entropy(emissions, tags, reduce=False)
        # shape: (batch_size, seq_length)
        entropy = entropy.view(batch_size, -1)
        # shape: (batch_size,)
        return torch.sum(entropy * mask.float(), dim=-1)

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
