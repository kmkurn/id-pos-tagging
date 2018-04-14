from typing import List, Tuple
import itertools
import os
import re

from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import BlanklineTokenizer, RegexpTokenizer


class CorpusReader(TaggedCorpusReader):
    DIGITS = re.compile(r'\d+')

    def __init__(self,
                 path: str,
                 encoding: str = 'utf8',
                 lower: bool = True,
                 replace_digits: bool = True,
                 ) -> None:
        self.__lower = lower
        self.__replace_digits = replace_digits

        word_tokenizer = RegexpTokenizer(r'\n', gaps=True)
        sent_tokenizer = BlanklineTokenizer()

        def para_block_reader(stream):
            return [stream.read()]

        super().__init__(
            *os.path.split(path), sep='\t', word_tokenizer=word_tokenizer,
            sent_tokenizer=sent_tokenizer, para_block_reader=para_block_reader,
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
