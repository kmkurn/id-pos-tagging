from collections import Counter
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple, Union
import warnings

import torch
import torch.nn as nn

from embedding import Embedding, EmbeddingWithPretrained
from models.neural import BiLSTMEmbedder, CRFTagger, Concatenate, ContextWindow, \
    EmissionScorer, GreedyTagger

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


def make_neural_tagger(
        num_words: int,
        num_tags: int,
        num_prefixes: Optional[Sequence[int]] = None,
        num_suffixes: Optional[Sequence[int]] = None,
        word_embedding_size: int = 100,
        prefix_embedding_size: Union[Sequence[int], int] = 20,
        suffix_embedding_size: Union[Sequence[int], int] = 20,
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

    final_emb = Concatenate(nn.ModuleList(embs))
    if use_lstm:
        final_emb = BiLSTMEmbedder(final_emb, emb_sz, hidden_size, padding_idx=padding_idx)
        emb_sz = 2 * hidden_size
    scorer = EmissionScorer(
        num_tags, final_emb, emb_sz, hidden_size=hidden_size, dropout=dropout)
    cls = CRFTagger if use_crf else GreedyTagger
    return cls(scorer, padding_idx=padding_idx)
