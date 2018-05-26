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
    max_iter = 100


@ex.named_config
def tuned_on_fold1():
    c2 = 0.002308
    seed = 615782537
    window = 0


@ex.named_config
def tuned_on_fold2():
    c2 = 0.117544
    seed = 687109176
    window = 0


@ex.named_config
def tuned_on_fold3():
    c2 = 0.004449
    seed = 558825499
    window = 0


@ex.named_config
def tuned_on_fold4():
    c2 = 0.262368
    seed = 915371814
    window = 1


@ex.named_config
def tuned_on_fold5():
    c2 = 0.186025
    seed = 414672713
    window = 1


nlp = spacy.blank('id')  # load once b/c this is slow


@ex.capture
def extract_crf_features(sent, window=2, use_prefix=True, use_suffix=True, use_wordshape=False):
    assert window >= 0, 'window cannot be negative'

    prefixes_2 = [w[:2] for w in sent]  # first 2 chars
    prefixes_3 = [w[:3] for w in sent]  # first 3 chars
    suffixes_2 = [w[-2:] for w in sent]  # last 2 chars
    suffixes_3 = [w[-3:] for w in sent]  # last 3 chars
    doc = Doc(nlp.vocab, words=sent)
    shapes = [doc[i].shape_ for i in range(len(sent))]  # wordshape

    sent = ['<s>'] + sent + ['</s>']
    prefixes_2 = ['<s>'] + prefixes_2 + ['</s>']
    prefixes_3 = ['<s>'] + prefixes_3 + ['</s>']
    suffixes_2 = ['<s>'] + suffixes_2 + ['</s>']
    suffixes_3 = ['<s>'] + suffixes_3 + ['</s>']
    shapes = ['<s>'] + shapes + ['</s>']

    pad_token = '<pad>'

    for i in range(1, len(sent) - 1):
        fs = {'w[0]': sent[i]}
        if use_prefix:
            fs['pref-2[0]'] = prefixes_2[i]
            fs['pref-3[0]'] = prefixes_3[i]
        if use_suffix:
            fs['suff-2[0]'] = suffixes_2[i]
            fs['suff-3[0]'] = suffixes_3[i]
        if use_wordshape:
            fs['shape[0]'] = shapes[i]

        # Contextual features
        for d in range(1, window + 1):
            fs[f'w[-{d}]'] = sent[i - d] if i - d >= 0 else pad_token
            fs[f'w[+{d}]'] = sent[i + d] if i + d < len(sent) else pad_token
            if use_prefix:
                fs[f'pref-2[-{d}]'] = prefixes_2[i - d] if i - d >= 0 else pad_token
                fs[f'pref-3[-{d}]'] = prefixes_3[i - d] if i - d >= 0 else pad_token
                fs[f'pref-2[+{d}]'] = prefixes_2[i + d] if i + d < len(sent) else pad_token
                fs[f'pref-3[+{d}]'] = prefixes_3[i + d] if i + d < len(sent) else pad_token
            if use_suffix:
                fs[f'suff-2[-{d}]'] = suffixes_2[i - d] if i - d >= 0 else pad_token
                fs[f'suff-3[-{d}]'] = suffixes_3[i - d] if i - d >= 0 else pad_token
                fs[f'suff-2[+{d}]'] = suffixes_2[i + d] if i + d < len(sent) else pad_token
                fs[f'suff-3[+{d}]'] = suffixes_3[i + d] if i + d < len(sent) else pad_token
            if use_wordshape:
                fs[f'shape[-{d}]'] = shapes[i - d] if i - d >= 0 else pad_token
                fs[f'shape[+{d}]'] = shapes[i + d] if i + d < len(sent) else pad_token

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
