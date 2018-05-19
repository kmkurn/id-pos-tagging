from typing import List, Tuple
import itertools
import os
import re
import sys

from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import BlanklineTokenizer, RegexpTokenizer
from pycrfsuite import Trainer
from sacred.observers import MongoObserver

SACRED_OBSERVE_FILES = os.getenv('SACRED_OBSERVE_FILES', 'false').lower() == 'true'


class CorpusReader(TaggedCorpusReader):
    DIGITS = re.compile(r'\d+')
    WORD_TOK = RegexpTokenizer(r'\n', gaps=True)
    SENT_TOK = BlanklineTokenizer()

    def __init__(
            self,
            path: str,
            encoding: str = 'utf8',
            max_sent_len: int = -1,
    ) -> None:
        self.__max_sent_len = max_sent_len

        def para_block_reader(stream):
            return [stream.read()]

        super().__init__(
            *os.path.split(path),
            sep='\t',
            word_tokenizer=self.WORD_TOK,
            sent_tokenizer=self.SENT_TOK,
            para_block_reader=para_block_reader,
            encoding=encoding)

    def paras(self) -> List[List[List[str]]]:
        paras = []
        for para in super().paras():
            sents = []
            for sent in para:
                if self.__max_sent_len != -1 and len(sent) > self.__max_sent_len:
                    continue
                sents.append(sent)
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
                tagged_sents.append(tagged_sent)
            tagged_paras.append(tagged_sents)
        return tagged_paras

    def tagged_sents(self) -> List[List[Tuple[str, str]]]:
        return list(itertools.chain.from_iterable(self.tagged_paras()))

    def tagged_words(self) -> List[Tuple[str, str]]:
        return list(itertools.chain.from_iterable(self.tagged_sents()))

    @classmethod
    def to_sents(cls, text: str) -> List[List[str]]:
        return [[word for word in cls.WORD_TOK.tokenize(sent)]
                for sent in cls.SENT_TOK.tokenize(text)]  # yapf: disable


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


def setup_mongo_observer(ex):
    mongo_url = os.getenv('SACRED_MONGO_URL')
    db_name = os.getenv('SACRED_DB_NAME')
    if mongo_url is not None and db_name is not None:
        ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


def run_predict(make_predictions):
    text = sys.stdin.read()
    sents = CorpusReader.to_sents(text)
    pred_labels = make_predictions(sents)
    index = 0
    for sent in sents:
        for word in sent:
            tag = pred_labels[index]
            print(f'{word}\t{tag}')
            index += 1
        print()


def separate_tagged_sents(tagged_sents):
    sents, tags = [], []
    for ts in tagged_sents:
        words, tags_ = zip(*ts)
        words, tags_ = list(words), list(tags_)
        sents.append(words)
        tags.append(tags_)
    return sents, tags
