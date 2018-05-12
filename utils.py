from typing import List, Tuple
import itertools
import os
import re

from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import BlanklineTokenizer, RegexpTokenizer
from pycrfsuite import Trainer


class CorpusReader(TaggedCorpusReader):
    DIGITS = re.compile(r'\d+')

    def __init__(
            self,
            path: str,
            encoding: str = 'utf8',
            lower: bool = True,
            replace_digits: bool = True,
            max_sent_len: int = -1,
    ) -> None:
        self.__lower = lower
        self.__replace_digits = replace_digits
        self.__max_sent_len = max_sent_len

        word_tokenizer = RegexpTokenizer(r'\n', gaps=True)
        sent_tokenizer = BlanklineTokenizer()

        def para_block_reader(stream):
            return [stream.read()]

        super().__init__(
            *os.path.split(path),
            sep='\t',
            word_tokenizer=word_tokenizer,
            sent_tokenizer=sent_tokenizer,
            para_block_reader=para_block_reader,
            encoding=encoding)

    def _preprocess_word(self, word: str) -> str:
        if self.__lower:
            word = word.lower()
        if self.__replace_digits:
            word = self.DIGITS.sub('0', word)
        return word

    def paras(self) -> List[List[List[str]]]:
        paras = []
        for para in super().paras():
            sents = []
            for sent in para:
                if self.__max_sent_len != -1 and len(sent) > self.__max_sent_len:
                    continue
                words = [self._preprocess_word(word) for word in sent]
                sents.append(words)
            paras.append(sents)
        return paras

    def sents(self) -> List[List[str]]:
        return list(itertools.chain.from_iterable(self.paras()))

    def words(self) -> List[str]:
        return list(itertools.chain.from_iterable(self.sents()))

    def tagged_paras(self) -> List[List[List[Tuple[str, str]]]]:
        tagged_paras = []
        for tagged_para in super().tagged_paras():
            tagged_sents = []
            for tagged_sent in tagged_para:
                if self.__max_sent_len != -1 and len(tagged_sent) > self.__max_sent_len:
                    continue
                tagged_words = []
                for word, tag in tagged_sent:
                    tagged_words.append((self._preprocess_word(word), tag))
                tagged_sents.append(tagged_words)
            tagged_paras.append(tagged_sents)
        return tagged_paras

    def tagged_sents(self) -> List[List[Tuple[str, str]]]:
        return list(itertools.chain.from_iterable(self.tagged_paras()))

    def tagged_words(self) -> List[Tuple[str, str]]:
        return list(itertools.chain.from_iterable(self.tagged_sents()))


class SacredAwarePycrfsuiteTrainer(Trainer):
    def __init__(self, run, *args, **kwargs):
        self.__run = run
        super().__init__(*args, **kwargs)

    def on_iteration(self, log, info):
        self.__run.log_scalar('loss(train)', info.get('loss'))
        for metric in 'precision recall f1'.split():
            # Overall
            attr = f'avg_{metric}'
            if attr in info:
                self.__run.log_scalar(f'{metric}(dev)', info[attr])
            # Per label
            for label, score in info['scores'].items():
                self.__run.log_scalar(f'{metric}(dev, {label})', getattr(score, metric))
        super().on_iteration(log, info)
