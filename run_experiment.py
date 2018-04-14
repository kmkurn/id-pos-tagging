#!/usr/bin/env python

import json
import os
import typing

from pycrfsuite import ItemSequence, Tagger, Trainer
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from utils import CorpusReader


ex = Experiment()

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default_conf():
    # file encoding
    encoding = 'utf8'
    # model name
    model_name = 'crf'

    if model_name == 'crf':
        # context window size
        window = 2
        # whether to lowercase the words
        lower = True
        # whether to replace digits with zeros
        replace_digits = True
        # whether to use prefix features
        use_prefix = True
        # whether to use suffix features
        use_suffix = True
        # features occurring less than this number will be discarded
        features_minfreq = 1
        # L2 regularization coefficient
        c2 = 1.0
        # maximum number of iterations
        max_iter = 2**31 - 1

    # path to train corpus
    train_path = 'train.tsv'
    # path to dev corpus
    dev_path = None
    # path to test corpus (only evaluate)
    test_path = 'test.tsv'
    # path to model file
    model_path = 'model'
    # where to serialize evaluation result
    eval_path = None
    # where to save the confusion matrix
    cm_path = None


@ex.capture
def read_corpus(path, _log, _run, name='train', encoding='utf-8', lower=True, replace_digits=True):
    _log.info(f'Reading {name} corpus from %s', path)
    _run.add_resource(path)
    return CorpusReader(path, encoding=encoding, lower=lower, replace_digits=replace_digits)


@ex.capture
def extract_crf_features(sent: typing.Sequence[str], window=2, use_prefix=True, use_suffix=True):
    if window < 0:
        raise ValueError('window cannot be negative, got', window)

    for i in range(len(sent)):
        fs = {'w[0]': sent[i]}
        for d in range(1, window + 1):
            fs[f'w[-{d}]'] = sent[i - d] if i - d >= 0 else '<s>'
            fs[f'w[+{d}]'] = sent[i + d] if i + d < len(sent) else '</s>'
        if use_prefix:
            fs['pref-2'] = sent[i][:2]  # first 2 chars
            fs['pref-3'] = sent[i][:3]  # first 3 chars
        if use_suffix:
            fs['suff-2'] = sent[i][-2:]  # last 2 chars
            fs['suff-3'] = sent[i][-3:]  # last 3 chars
        yield fs


@ex.capture
def make_predictions(reader, model_name, model_path, _log, _run):
    if model_name != 'crf':
        raise NotImplementedError

    _log.info('Loading model from %s', model_path)
    _run.add_resource(model_path)
    tagger = Tagger()
    tagger.open(model_path)

    _log.info('Extracting features from test corpus')
    itemseq = ItemSequence([fs for sent in reader.sents() for fs in extract_crf_features(sent)])

    _log.info('Making predictions with the model')
    return tagger.tag(itemseq)


@ex.capture
def evaluate_fully(gold_labels, pred_labels, eval_path, _log, _run, result=None):
    if result is None:
        result = {}

    all_labels = list(set(gold_labels + pred_labels))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=all_labels)
    for label, p, r, f in zip(all_labels, prec, rec, f1):
        result[label] = {'P': p, 'R': r, 'F1': f}
    _log.info('Saving the full evaluation result to %s', eval_path)
    with open(eval_path, 'w') as f:
        json.dump(result, f, indent=2)
    _run.add_artifact(eval_path)


@ex.capture
def plot_confusion_matrix(gold_labels, pred_labels, cm_path, _log, _run):
    all_labels = list(set(gold_labels + pred_labels))
    _log.info('Saving the confusion matrix to %s', cm_path)
    cm = confusion_matrix(gold_labels, pred_labels, labels=all_labels)
    cm = cm / cm.sum(axis=1).reshape(-1, 1)
    sns.set()
    sns.heatmap(
        cm, vmin=0, vmax=1, xticklabels=all_labels, yticklabels=all_labels, cmap='YlGnBu')
    plt.savefig(cm_path)
    _run.add_artifact(cm_path)


@ex.command
def train(_log, model_name, train_path, model_path, dev_path=None, features_minfreq=1, c2=1.0,
          max_iter=2**31 - 1):
    """Train a model."""
    if model_name != 'crf':
        raise NotImplementedError

    train_reader = read_corpus(train_path)
    _log.info('Extracting features from train corpus')
    train_itemseq = ItemSequence([
        fs for sent in train_reader.sents() for fs in extract_crf_features(sent)])
    train_labels = [tag for _, tag in train_reader.tagged_words()]

    params = {'feature.minfreq': features_minfreq, 'c2': c2, 'max_iterations': max_iter}
    trainer = Trainer(algorithm='lbfgs', params=params)
    trainer.append(train_itemseq, train_labels, group=0)

    if dev_path is not None:
        dev_reader = read_corpus(dev_path, name='dev')
        _log.info('Extracting features from dev corpus')
        dev_itemseq = ItemSequence([
            fs for sent in dev_reader.sents() for fs in extract_crf_features(sent)])
        dev_labels = [tag for _, tag in dev_reader.tagged_words()]
        trainer.append(dev_itemseq, dev_labels, group=1)

    _log.info('Begin training; saving model to %s', model_path)
    trainer.train(model_path, holdout=1)


@ex.command
def predict(_log, model_name, test_path):
    """Make predictions using a trained model."""
    if model_name != 'crf':
        raise NotImplementedError

    reader = read_corpus(test_path, name='test')
    pred_labels = make_predictions(reader)
    index = 0
    for sent in reader.sents():
        for word in sent:
            tag = pred_labels[index]
            print(f'{word}\t{tag}')
            index += 1
        print()


@ex.automain
def evaluate(model_name, test_path, eval_path=None, cm_path=None):
    """Evaluate a trained model."""
    if model_name != 'crf':
        raise NotImplementedError

    reader = read_corpus(test_path, name='test')
    gold_labels = [tag for _, tag in reader.tagged_words()]
    pred_labels = make_predictions(reader)
    result = {'overall_acc': accuracy_score(gold_labels, pred_labels)}

    if eval_path is not None:
        evaluate_fully(gold_labels, pred_labels, result=result)
    if cm_path is not None:
        plot_confusion_matrix(gold_labels, pred_labels)

    return result['overall_acc']
