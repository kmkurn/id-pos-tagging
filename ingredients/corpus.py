from sacred import Ingredient

from utils import SACRED_OBSERVE_FILES, CorpusReader

ing = Ingredient('corpus')


@ing.config
def cfg():
    # file encoding
    encoding = 'utf8'
    # path to train corpus
    train = 'train.tsv'
    # path to dev corpus
    dev = None
    # path to test corpus
    test = 'test.tsv'


@ing.capture
def read_corpus(path, _log, _run, name='train', encoding='utf-8', max_sent_len=-1):
    _log.info(f'Reading {name} corpus from %s', path)
    reader = CorpusReader(path, encoding=encoding, max_sent_len=max_sent_len)
    if SACRED_OBSERVE_FILES:
        _run.add_resource(path)
    return reader


@ing.capture
def read_train_corpus(train, **kwargs):
    return read_corpus(train, **kwargs)


@ing.capture
def read_dev_corpus(dev, **kwargs):
    return read_corpus(dev, name='dev', **kwargs) if dev is not None else None


@ing.capture
def read_test_corpus(test, **kwargs):
    return read_corpus(test, name='test', **kwargs)
