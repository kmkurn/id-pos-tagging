#!/usr/bin/env python

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_corpus
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.preprocessing import ing as prep_ingredient, preprocess
from models.tagger import MemorizationTagger
from serialization import dump, load
from utils import SACRED_OBSERVE_FILES, run_predict, separate_tagged_sents, setup_mongo_observer

ingredients = [corpus_ingredient, eval_ingredient, prep_ingredient]
ex = Experiment(name='id-pos-tagging-memo-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # path to model file
    model_path = 'model'
    # context window size
    window = 2


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
    sents = preprocess(sents)
    _log.info('Making predictions with the model')
    return [tag for sent in sents for tag in model.predict(sent)]


@ex.command
def train(model_path, _log, _run, window=2):
    """Train a memorization model."""
    train_reader = read_train_corpus()
    sents, tags = separate_tagged_sents(train_reader.tagged_sents())
    sents = preprocess(sents)
    _log.info('Start training model')
    model = MemorizationTagger.train(sents, tags, window=window)
    _log.info('Saving model to %s', model_path)
    with open(model_path, 'w') as f:
        print(dump(model), file=f)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(model_path)


@ex.command(unobserved=True)
def predict():
    """Make predictions using a trained memorization model."""
    model = load_model()
    run_predict(lambda sents: make_preds(model, sents))


@ex.automain
def evaluate():
    """Evaluate a trained memorization model."""
    model = load_model()
    return run_evaluation(lambda sents: make_preds(model, sents))
