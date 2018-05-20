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
    # maximum allowed sentence length or -1 for no limit
    max_sent_len = -1


@ing.capture
def read_corpus(path, _log, _run, name='train', encoding='utf-8', max_sent_len=-1):
    _log.info(f'Reading {name} corpus from %s', path)
    reader = CorpusReader(path, encoding=encoding, max_sent_len=max_sent_len)
    if SACRED_OBSERVE_FILES:
        _run.add_resource(path)
    return reader


@ing.capture
def read_train_corpus(train):
    return read_corpus(train)


@ing.capture
def read_dev_corpus(dev):
    return read_corpus(dev, name='dev') if dev is not None else None


@ing.capture
def read_test_corpus(test):
    return read_corpus(test, name='test')
