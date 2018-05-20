#!/usr/bin/env python

import spacy
from pycrfsuite import ItemSequence, Tagger
from sacred import Experiment
from spacy.tokens import Doc

from ingredients.corpus import ing as corpus_ingredient, read_train_corpus, read_dev_corpus
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.preprocessing import ing as prep_ingredient, preprocess
from utils import SACRED_OBSERVE_FILES, SacredAwarePycrfsuiteTrainer as Trainer, run_predict, \
    separate_tagged_sents, setup_mongo_observer

ingredients = [corpus_ingredient, eval_ingredient, prep_ingredient]
ex = Experiment(name='id-pos-tagging-crf-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # path to model file
    model_path = 'model'
    # context window size
    window = 2
    # whether to use prefix features
    use_prefix = True
    # whether to use suffix features
    use_suffix = True
    # whether to use wordshape features
    use_wordshape = False
    # features occurring less than this number will be discarded
    min_freq = 1
    # L2 regularization coefficient
    c2 = 1.0
    # maximum number of iterations
    max_iter = 2**31 - 1


nlp = spacy.blank('id')  # load once b/c this is slow


@ex.capture
def extract_crf_features(sent, window=2, use_prefix=True, use_suffix=True, use_wordshape=False):
    assert window >= 0, 'window cannot be negative'

    doc = Doc(nlp.vocab, words=sent)
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
        if use_wordshape:
            fs['shape'] = doc[i].shape_
        yield fs


@ex.capture
def make_crf_trainer(_run, min_freq=1, c2=1.0, max_iter=2**31 - 1):
    params = {'feature.minfreq': min_freq, 'c2': c2, 'max_iterations': max_iter}
    return Trainer(_run, algorithm='lbfgs', params=params)


@ex.capture
def load_model(model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    tagger = Tagger()
    tagger.open(model_path)
    if SACRED_OBSERVE_FILES:
        _run.add_resource(model_path)
    return tagger


@ex.capture
def make_preds(tagger, sents, _log):
    sents = preprocess(sents)
    _log.info('Extracting features')
    itemseq = ItemSequence([fs for sent in sents for fs in extract_crf_features(sent)])

    _log.info('Making predictions with the model')
    return tagger.tag(itemseq)


@ex.command
def train(model_path, _log, _run, window=2):
    """Train a CRF model."""
    train_reader = read_train_corpus()
    sents, tags = separate_tagged_sents(train_reader.tagged_sents())
    sents = preprocess(sents)
    _log.info('Extracting features from train corpus')
    train_itemseq = ItemSequence([fs for sent in sents for fs in extract_crf_features(sent)])
    train_labels = [tag for tags_ in tags for tag in tags_]

    trainer = make_crf_trainer()
    trainer.append(train_itemseq, train_labels, group=0)

    dev_reader = read_dev_corpus()
    if dev_reader is not None:
        _log.info('Extracting features from dev corpus')
        sents, tags = separate_tagged_sents(dev_reader.tagged_sents())
        sents = preprocess(sents)
        dev_itemseq = ItemSequence([fs for sent in sents for fs in extract_crf_features(sent)])
        dev_labels = [tag for tags_ in tags for tag in tags_]
        trainer.append(dev_itemseq, dev_labels, group=1)

    _log.info('Begin training; saving model to %s', model_path)
    holdout = -1 if dev_reader is None else 1
    trainer.train(model_path, holdout=holdout)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(model_path)


@ex.command(unobserved=True)
def predict():
    """Make predictions using a trained CRF model."""
    model = load_model()
    run_predict(lambda sents: make_preds(model, sents))


@ex.automain
def evaluate():
    """Evaluate a trained CRF model."""
    model = load_model()
    return run_evaluation(lambda sents: make_preds(model, sents))
