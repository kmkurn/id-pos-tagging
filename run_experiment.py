#!/usr/bin/env python

import enum
import json
import os
import typing

from pycrfsuite import ItemSequence, Tagger, Trainer
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from spacy.tokens import Doc
from torchtext.data import BucketIterator, Dataset, Example, Field
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchnet as tnt

from models import FeedforwardTagger
from utils import CorpusReader


ex = Experiment()

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


class ModelName(enum.Enum):
    # Conditional random field (Pisceldo et al., 2009)
    CRF = 'crf'
    # Feedforward neural network (Abka, 2016)
    FF = 'feedforward'


@ex.config
def default_conf():
    # file encoding
    encoding = 'utf8'
    # model name [crf, feedforward]
    model_name = ModelName.CRF.value
    # context window size
    window = 2
    # whether to lowercase the words
    lower = True
    # whether to replace digits with zeros
    replace_digits = True

    if ModelName(model_name) is ModelName.CRF:
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
    else:
        # size of word embedding
        word_embedding_size = 100
        # size of hidden layer
        hidden_size = 100
        # dropout rate
        dropout = 0.5
        # batch size
        batch_size = 16
        # GPU device, or -1 for CPU
        device = -1
        # print training log every this iterations
        print_every = 10
        # maximum number of training epochs
        max_epochs = 50
        # wait for this number of times reducing LR before early stopping
        stopping_patience = 5
        # reduce LR when accuracy does not improve after this number of epochs
        scheduler_patience = 2
        # tolerance for comparing accuracy for early stopping
        tol = 0.01
        # whether to print to stdout when LR is reduced
        scheduler_verbose = False

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
def get_model_name_enum(model_name):
    return ModelName(model_name)


@ex.capture
def read_corpus(path, _log, _run, name='train', encoding='utf-8', lower=True, replace_digits=True):
    _log.info(f'Reading {name} corpus from %s', path)
    _run.add_resource(path)
    return CorpusReader(path, encoding=encoding, lower=lower, replace_digits=replace_digits)


nlp = spacy.blank('id')  # load once b/c this is slow


@ex.capture
def extract_crf_features(sent: typing.Sequence[str], window=2, use_prefix=True, use_suffix=True,
                         use_wordshape=False):
    if window < 0:
        raise ValueError('window cannot be negative, got', window)

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
def make_crf_trainer(min_freq=1, c2=1.0, max_iter=2**31 - 1):
    params = {'feature.minfreq': min_freq, 'c2': c2, 'max_iterations': max_iter}
    return Trainer(algorithm='lbfgs', params=params)


@ex.capture
def train_crf_model(train_path, model_path, _log, dev_path=None):
    train_reader = read_corpus(train_path)
    _log.info('Extracting features from train corpus')
    train_itemseq = ItemSequence([
        fs for sent in train_reader.sents() for fs in extract_crf_features(sent)])
    train_labels = [tag for _, tag in train_reader.tagged_words()]

    trainer = make_crf_trainer()
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


@ex.capture
def prepare_fields(_log):
    _log.info('Prepare fields')
    WORDS = Field(  # no `lower` because it's done in `CorpusReader`
        batch_first=True, include_lengths=True)
    TAGS = Field(batch_first=True, unk_token=None)
    return WORDS, TAGS


@ex.capture
def make_dataset(path, fields, _log, name='train'):
    if len(fields) not in (2, 3):
        raise ValueError('fields should contain 2 or 3 elements')

    _log.info('Creating %s dataset', name)
    if isinstance(path, str):
        reader = read_corpus(path, name=name)
    else:
        reader = path
    examples = []
    for id_, tagged_sent in enumerate(reader.tagged_sents()):
        words, tags = zip(*tagged_sent)
        if len(fields) == 3:
            example = Example.fromlist([id_, words, tags], fields)
        else:
            assert len(fields) == 2
            example = Example.fromlist([words, tags], fields)
        examples.append(example)
    return Dataset(examples, fields)


@ex.capture
def build_vocab(fields: typing.Sequence[typing.Tuple[str, Field]],
                datasets: typing.Sequence[Dataset],
                _log,
                ) -> None:
    _log.info('Building vocabulary')
    for name, field in fields:
        field.build_vocab(*datasets)
        _log.info('Found %d %s', len(field.vocab), name)


@ex.capture
def make_feedforward_model(num_words, num_tags, _log, word_embedding_size=100, window=2,
                           hidden_size=100, dropout=0.5, padding_idx=0):
    _log.info('Creating the feedforward model')
    return FeedforwardTagger(
        num_words, num_tags, word_embedding_size=word_embedding_size, window=window,
        hidden_size=hidden_size, dropout=dropout, padding_idx=padding_idx)


@ex.capture
def train_feedforward_model(train_path, model_path, _log, dev_path=None, batch_size=16,
                            device=-1, print_every=10, max_epochs=20, stopping_patience=5,
                            scheduler_patience=2, tol=0.01, scheduler_verbose=False):
    WORDS, TAGS = prepare_fields()
    fields = [('words', WORDS), ('tags', TAGS)]

    train_dataset = make_dataset(train_path, fields)
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    train_iter = BucketIterator(
        train_dataset, batch_size, sort_key=sort_key, device=device, repeat=False)
    dev_iter = None

    if dev_path is not None:
        dev_dataset = make_dataset(dev_path, fields, name='dev')
        dev_iter = BucketIterator(
            dev_dataset, batch_size, sort_key=sort_key, device=device, train=False)

    build_vocab(fields, (train_dataset,))
    assert hasattr(WORDS, 'vocab')
    assert hasattr(TAGS, 'vocab')

    num_words, num_tags = len(WORDS.vocab), len(TAGS.vocab)
    model = make_feedforward_model(
        num_words, num_tags, padding_idx=WORDS.vocab.stoi[WORDS.pad_token])
    if device >= 0:
        model.cuda(device)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=scheduler_patience, threshold=tol,
        threshold_mode='abs', verbose=scheduler_verbose)

    engine = tnt.engine.Engine()
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.AverageValueMeter()
    speed_meter = tnt.meter.AverageValueMeter()
    train_timer = tnt.meter.TimeMeter(None)
    epoch_timer = tnt.meter.TimeMeter(None)
    batch_timer = tnt.meter.TimeMeter(None)

    def net(minibatch):
        # shape: (batch_size, seq_length)
        words, _ = minibatch.words
        # shape: (batch_size, seq_length, num_tags)
        outputs = model(words)
        # shape: (batch_size * seq_length, num_tags)
        outputs = outputs.view(-1, num_tags)
        # shape: (batch_size * seq_length,)
        tags = minibatch.tags.view(-1)
        # shape: (1,)
        loss = F.cross_entropy(outputs, tags, ignore_index=TAGS.vocab.stoi[TAGS.pad_token])
        # shape: (batch_size, seq_length)
        predictions = model.predict(words)

        return loss, predictions

    def reset_meters():
        loss_meter.reset()
        acc_meter.reset()
        speed_meter.reset()

    def save_model():
        _log.info('Saving model weights to %s', model_path)
        torch.save(model.state_dict(), model_path)

    def on_start(state):
        if state['train']:
            save_model()
            state.update(dict(best_acc=-float('inf'), num_bad_epochs=0))
            _log.info('Start training')
            train_timer.reset()
        else:
            reset_meters()
            epoch_timer.reset()
            model.eval()

    def on_start_epoch(state):
        _log.info('Starting epoch %s', state['epoch'] + 1)
        reset_meters()
        epoch_timer.reset()
        model.train()

    def on_sample(state):
        batch_timer.reset()

    def on_forward(state):
        loss_meter.add(state['loss'].data[0])

        # shape: (batch_size, seq_length)
        golds = state['sample'].tags
        # shape: (batch_size, seq_length)
        predictions = state['output']
        # shape: (batch_size,)
        _, lengths = state['sample'].words
        # Compute accuracy
        for gold, pred, length in zip(golds, predictions, lengths):
            gold, pred = gold[:length], pred[:length]
            acc = accuracy_score(gold.data.numpy(), pred.data.numpy())
            acc_meter.add(acc)

        elapsed_time = batch_timer.value()
        speed = lengths.size(0) / elapsed_time
        speed_meter.add(speed)

        if state['train'] and (state['t'] + 1) % print_every == 0:
            epoch = (state['t'] + 1) / len(state['iterator'])
            _log.info(
                'Epoch %.2f (%.2fms): %.2f samples/s | loss %.4f | acc %.2f',
                epoch, 1000 * elapsed_time, speed_meter.value()[0], loss_meter.value()[0],
                acc_meter.value()[0])

    def on_end_epoch(state):
        elapsed_time = epoch_timer.value()
        _log.info(
            'Epoch %d done (%.2fs): %.2f samples/s | loss %.4f | acc %.2f',
            state['epoch'], epoch_timer.value(), speed_meter.value()[0],
            loss_meter.value()[0], acc_meter.value()[0])

        if dev_iter is not None:
            _log.info('Evaluating on dev corpus')
            engine.test(net, dev_iter)
            dev_acc = acc_meter.value()[0]
            _log.info('Result on dev corpus (%.2fs): %.2f samples/s | loss %.4f | acc %.2f',
                      epoch_timer.value(), speed_meter.value()[0], loss_meter.value()[0],
                      dev_acc)

            scheduler.step(dev_acc, epoch=state['epoch'])
            if dev_acc >= state['best_acc'] + tol:
                _log.info('New best result on dev corpus')
                state.update(dict(best_acc=dev_acc, num_bad_epochs=0))
                save_model()
            else:
                state['num_bad_epochs'] += 1
                if state['num_bad_epochs'] >= stopping_patience * (scheduler_patience + 1):
                    num_reduction = state['num_bad_epochs'] // (scheduler_patience + 1)
                    _log.info(
                        f"No improvements after {num_reduction} LR reductions, stopping early")
                    state['maxepoch'] = -1  # force training loop to stop

    def on_end(state):
        if state['train']:
            _log.info('Training done in %.2fs', train_timer.value())

    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end'] = on_end

    try:
        engine.train(net, train_iter, max_epochs, optimizer)
    except KeyboardInterrupt:
        _log.info('Training interrupted, aborting')
        save_model()


@ex.capture
def predict_crf(reader, model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    _run.add_resource(model_path)
    tagger = Tagger()
    tagger.open(model_path)

    _log.info('Extracting features from test corpus')
    itemseq = ItemSequence([fs for sent in reader.sents() for fs in extract_crf_features(sent)])

    _log.info('Making predictions with the model')
    return tagger.tag(itemseq)


@ex.capture
def predict_feedforward(reader, train_path, model_path, _log, _run, device=-1, batch_size=16):
    WORDS, TAGS = prepare_fields()
    IDS = Field(sequential=False, use_vocab=False)
    fields = [('index', IDS), ('words', WORDS), ('tags', TAGS)]

    train_dataset = make_dataset(train_path, fields[1:])
    build_vocab(fields[1:], (train_dataset,))

    num_words, num_tags = len(WORDS.vocab), len(TAGS.vocab)
    model = make_feedforward_model(
        num_words, num_tags, padding_idx=WORDS.vocab.stoi[WORDS.pad_token])

    _log.info('Loading model weights from %s', model_path)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    if device >= 0:
        model.cuda(device)
    model.eval()

    test_dataset = make_dataset(reader, fields, name='test')
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    test_iter = BucketIterator(
        test_dataset, batch_size, sort_key=sort_key, device=device, train=False)

    indexed_predictions = []
    for minibatch in test_iter:
        # shape: (batch_size, seq_length), (batch_size,)
        words, lengths = minibatch.words
        # shape: (batch_size, seq_length)
        predictions = model.predict(words)
        # shape: (batch_size,)
        index = minibatch.index

        for i, pred, length in zip(index, predictions, lengths):
            pred = pred[:length]
            indexed_predictions.append((i.data[0], pred.data))

    indexed_predictions.sort()
    return [TAGS.vocab.itos[pred] for _, pred_sent in indexed_predictions
            for pred in pred_sent]


@ex.capture
def make_predictions(reader):
    model_name = get_model_name_enum()
    if model_name is ModelName.CRF:
        return predict_crf(reader)
    return predict_feedforward(reader)


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
def set_random_seed(_seed):
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)


@ex.command
def train():
    """Train a model."""
    set_random_seed()
    if get_model_name_enum() is ModelName.CRF:
        train_crf_model()
    else:
        train_feedforward_model()


@ex.command
def predict(_log, test_path):
    """Make predictions using a trained model."""
    set_random_seed()
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
def evaluate(test_path, eval_path=None, cm_path=None):
    """Evaluate a trained model."""
    set_random_seed()
    reader = read_corpus(test_path, name='test')
    gold_labels = [tag for _, tag in reader.tagged_words()]
    pred_labels = make_predictions(reader)
    result = {'overall_acc': accuracy_score(gold_labels, pred_labels)}

    if eval_path is not None:
        evaluate_fully(gold_labels, pred_labels, result=result)
    if cm_path is not None:
        plot_confusion_matrix(gold_labels, pred_labels)

    return result['overall_acc']
