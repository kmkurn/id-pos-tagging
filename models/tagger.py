from collections import Counter
from contextlib import contextmanager
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from torch.autograd import Variable as Var
from torchcrf import CRF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.components import BiLSTMEmbedder, CNNEncoder, Concatenate, ContextWindow, \
    TimeDistributed
from models.embedding import Embedding, EmbeddingWithPretrained

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
    def train(
            cls,
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


class EmissionScorer(nn.Module):
    def __init__(
            self,
            num_tags: int,
            embedder: nn.Module,
            embedder_size: int,
            hidden_size: int = 100,
            dropout: float = 0.5,
    ) -> None:
        super(EmissionScorer, self).__init__()
        self.embedder = embedder
        self.ff = nn.Sequential(
            nn.Linear(embedder_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_tags),
        )
        self.reset_parameters()

    @property
    def num_tags(self) -> int:
        return self.ff[-1].out_features

    def reset_parameters(self) -> None:
        if hasattr(self.embedder, 'reset_parameters'):
            self.embedder.reset_parameters()
        init.xavier_uniform(self.ff[0].weight, gain=init.calculate_gain('tanh'))
        init.constant(self.ff[0].bias, 0.)
        init.xavier_uniform(self.ff[-1].weight, gain=init.calculate_gain('linear'))
        init.constant(self.ff[-1].bias, 0.)

    def forward(self, inputs: Sequence[Var]) -> Var:
        assert all(i.dim() >= 2 for i in inputs)

        # shape: (batch_size, seq_length, size)
        embedded = self.embedder(inputs)
        # shape: (batch_size, seq_length, num_tags)
        emissions = self.ff(embedded)

        return emissions


class GreedyTagger(nn.Module):
    def __init__(self, scorer: EmissionScorer, padding_idx: int = 0) -> None:
        super(GreedyTagger, self).__init__()
        self.scorer = scorer
        self.padding_idx = padding_idx
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scorer.reset_parameters()

    def forward(self, inputs: Sequence[Var], tags: Var) -> Var:
        assert all(i.dim() in (2, 3) for i in inputs)
        assert all(i.size()[:2] == tags.size() for i in inputs)

        batch_size = tags.size(0)

        # shape: (batch_size, seq_length, num_tags)
        emissions = self.scorer(inputs)
        # shape: (batch_size * seq_length, num_tags)
        emissions = emissions.view(-1, emissions.size(-1))
        # shape: (batch_size * seq_length,)
        tags = tags.view(-1)
        # shape: (1,)
        loss = F.cross_entropy(emissions, tags, ignore_index=self.padding_idx)

        return loss

    def decode(self, inputs: Sequence[Var]) -> List[List[int]]:
        assert all(i.dim() in (2, 3) for i in inputs)

        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self.scorer(inputs)

        # shape: (batch_size, seq_length)
        _, best_tags = emissions.max(dim=-1)
        # shape: (batch_size, seq_length, N)
        cat = torch.cat([i.unsqueeze(-1) if i.dim() == 2 else i for i in inputs], dim=-1)
        # shape: (batch_size, seq_length)
        mask = torch.sum((cat != self.padding_idx).long(), dim=-1) != 0

        best_tags, mask = best_tags.data, mask.data
        result = []
        for best, mask_ in zip(best_tags, mask):
            length = mask_.long().sum()
            result.append(best[:length].tolist())
        return result


class CRFTagger(nn.Module):
    def __init__(self, scorer: EmissionScorer, padding_idx: int = 0) -> None:
        super(CRFTagger, self).__init__()
        self.scorer = scorer
        self.padding_idx = padding_idx
        self.crf = CRF(scorer.num_tags)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.scorer.reset_parameters()
        self.crf.reset_parameters()

    def forward(self, inputs: Sequence[Var], tags: Var) -> Var:
        assert all(i.dim() in (2, 3) for i in inputs)
        assert all(i.size()[:2] == tags.size() for i in inputs)

        # shape: (seq_length, batch_size)
        tags = tags.transpose(0, 1).contiguous()
        # shape: (batch_size, seq_length, N)
        cat = torch.cat([i.unsqueeze(-1) if i.dim() == 2 else i for i in inputs], dim=-1)
        # shape: (batch_size, seq_length)
        mask = torch.sum((cat != self.padding_idx).long(), dim=-1) != 0
        # shape: (seq_length, batch_size)
        mask = mask.transpose(0, 1).contiguous()

        # shape: (batch_size, seq_length, num_tags)
        emissions = self.scorer(inputs)
        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()
        # shape: (1,)
        loss = -self.crf(emissions, tags, mask=mask) / mask.float().sum()

        return loss

    def decode(self, inputs: Sequence[Var]) -> List[List[int]]:
        assert all(i.dim() in (2, 3) for i in inputs)

        # shape: (batch_size, seq_length, N)
        cat = torch.cat([i.unsqueeze(-1) if i.dim() == 2 else i for i in inputs], dim=-1)
        # shape: (batch_size, seq_length)
        mask = torch.sum((cat != self.padding_idx).long(), dim=-1) != 0
        # shape: (seq_length, batch_size)
        mask = mask.transpose(0, 1).contiguous()

        with evaluation(self):
            # shape: (batch_size, seq_length, num_tags)
            emissions = self.scorer(inputs)

        # shape: (seq_length, batch_size, num_tags)
        emissions = emissions.transpose(0, 1).contiguous()

        return self.crf.decode(emissions, mask=mask)


def make_neural_tagger(
        num_words: int,
        num_tags: int,
        num_prefixes: Optional[Sequence[int]] = None,
        num_suffixes: Optional[Sequence[int]] = None,
        num_chars: Optional[int] = None,
        word_embedding_size: int = 100,
        prefix_embedding_size: Union[Sequence[int], int] = 20,
        suffix_embedding_size: Union[Sequence[int], int] = 20,
        char_embedding_size: int = 50,
        num_char_filters: int = 100,
        filter_width: int = 3,
        window: int = 2,
        hidden_size: int = 100,
        dropout: float = 0.5,
        use_lstm: bool = False,
        use_crf: bool = False,
        padding_idx: int = 0,
        pretrained_embedding: Optional[torch.Tensor] = None,
) -> nn.Module:
    if use_lstm and window > 0:
        warnings.warn(
            "use_lstm=True but window > 0; it's recommended to set window=0 so the network "
            "architecture is more 'normal'")

    # Word features
    word_emb = nn.Sequential(
        EmbeddingWithPretrained(
            num_words,
            word_embedding_size,
            pretrained_embedding=pretrained_embedding,
            padding_idx=padding_idx,
            dropout=dropout),
        ContextWindow(window),
    )

    embs = [word_emb]
    emb_sz = (2 * window + 1) * word_embedding_size

    def embed_affixes(num_affixes, affix_emb_sz):
        if isinstance(affix_emb_sz, Sequence):
            if len(affix_emb_sz) != len(num_affixes):
                raise ValueError(
                    'affix embedding size should have length of', len(num_affixes), 'but got',
                    len(affix_emb_sz))
        else:
            assert isinstance(affix_emb_sz, int)
            affix_emb_sz = [affix_emb_sz] * len(num_affixes)

        nonlocal emb_sz
        for n, sz in zip(num_affixes, affix_emb_sz):
            emb = Embedding(n, sz, padding_idx=padding_idx, dropout=dropout)
            embs.append(emb)
            emb_sz += sz

    # Prefix features
    if num_prefixes is not None:
        embed_affixes(num_prefixes, prefix_embedding_size)

    # Suffix features
    if num_suffixes is not None:
        embed_affixes(num_suffixes, suffix_embedding_size)

    # Char features
    if num_chars is not None:
        char_emb = TimeDistributed(
            nn.Sequential(
                Embedding(
                    num_chars, char_embedding_size, padding_idx=padding_idx, dropout=dropout),
                CNNEncoder(
                    char_embedding_size,
                    num_filters=num_char_filters,
                    filter_width=filter_width),
            ))
        embs.append(char_emb)
        emb_sz += num_char_filters

    final_emb = Concatenate(nn.ModuleList(embs))
    if use_lstm:
        final_emb = nn.Sequential(final_emb, nn.Dropout(dropout))
        final_emb = BiLSTMEmbedder(final_emb, emb_sz, hidden_size, padding_idx=padding_idx)
        emb_sz = 2 * hidden_size
    scorer = EmissionScorer(
        num_tags, final_emb, emb_sz, hidden_size=hidden_size, dropout=dropout)
    cls = CRFTagger if use_crf else GreedyTagger
    return cls(scorer, padding_idx=padding_idx)


@contextmanager
def evaluation(model):
    training = model.training
    model.eval()
    yield
    model.train(training)
