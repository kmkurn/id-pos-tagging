#!/usr/bin/env python

from collections import Counter

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_corpus
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from serialization import dump, load
from utils import SACRED_OBSERVE_FILES, run_predict, setup_mongo_observer

ingredients = [corpus_ingredient, eval_ingredient]
ex = Experiment(name='id-pos-tagging-majority-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # path to model file
    model_path = 'model'


@ex.capture
def load_model(model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path) as f:
        model = load(f.read())
    if SACRED_OBSERVE_FILES:
        _run.add_resource(model_path)
    return model


@ex.capture
def make_preds(model, sents, _log):
    _log.info('Making predictions with the model')
    words = [word for sent in sents for word in sent]
    return [model['majority_tag']] * len(words)


@ex.command
def train(model_path, _log, _run):
    """Train a majority vote model."""
    train_reader = read_train_corpus()
    c = Counter(tag for _, tag in train_reader.tagged_words())
    majority_tag = c.most_common(n=1)[0][0]
    _log.info('Saving model to %s', model_path)
    with open(model_path, 'w') as f:
        print(dump({'majority_tag': majority_tag}), file=f)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(model_path)


@ex.command(unobserved=True)
def predict():
    """Make predictions using a trained majority vote model."""
    model = load_model()
    run_predict(lambda sents: make_preds(model, sents))


@ex.automain
def evaluate():
    """Evaluate a trained majority vote model."""
    model = load_model()
    return run_evaluation(lambda sents: make_preds(model, sents))
