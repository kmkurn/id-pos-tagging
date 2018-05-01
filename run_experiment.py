#!/usr/bin/env python

from collections import Counter, defaultdict
import enum
import json
import os
import pickle
import shutil

from pycrfsuite import ItemSequence, Tagger
from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from spacy.tokens import Doc
from torchtext.data import BucketIterator, Dataset, Example, Field
from torchtext.vocab import FastText
import dill
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import torch
import torch.optim as optim
import torchnet as tnt

from models import FeedforwardTagger, MemorizationTagger
from utils import CorpusReader, SacredAwarePycrfsuiteTrainer as Trainer


ex = Experiment(name='id-pos-tagging')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


class ModelName(enum.Enum):
    # Majority vote baseline
    MAJOR = 'majority'
    # Memorization baseline
    MEMO = 'memo'
    # Conditional random field (Pisceldo et al., 2009)
    CRF = 'crf'
    # Feedforward neural network (Abka, 2016)
    FF = 'feedforward'


@ex.config
def default_conf():
    # file encoding
    encoding = 'utf8'
    # model name [majority, crf, feedforward]
    model_name = ModelName.CRF.value
    # whether to lowercase the words
    lower = True
    # whether to replace digits with zeros
    replace_digits = True

    if ModelName(model_name) is ModelName.MAJOR:
        # path to model file
        model_path = 'model'
    elif ModelName(model_name) is ModelName.MEMO:
        # path to model file
        model_path = 'model'
        # context window size
        window = 0
    elif ModelName(model_name) is ModelName.CRF:
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
    else:
        # path to save directory (models and checkpoints will be saved here)
        save_dir = 'save_dir'
        # resume training from the checkpoint saved in this directory
        resume_from = None
        # words occurring fewer than this value will not be included in the vocab
        min_word_freq = 2
        # context window size
        window = 2
        # size of word embedding
        word_embedding_size = 100
        # size of hidden layer
        hidden_size = 100
        # dropout rate
        dropout = 0.5
        # whether to use CRF layer as output layer instead of softmax
        use_crf = False
        # learning rate
        lr = 0.001
        # batch size
        batch_size = 16
        # GPU device, or -1 for CPU
        device = 0 if torch.cuda.is_available() else -1
        # print training log every this iterations
        print_every = 10
        # maximum number of training epochs
        max_epochs = 50
        # wait for this number of times reducing LR before early stopping
        stopping_patience = 5
        # reduce LR when F1 score does not improve after this number of epochs
        scheduler_patience = 2
        # tolerance for comparing F1 score for early stopping
        tol = 0.01
        # whether to print to stdout when LR is reduced
        scheduler_verbose = False
        # use fasttext pretrained word embedding
        use_fasttext = False

    # path to train corpus (only train)
    train_path = 'train.tsv'
    # path to dev corpus (only train)
    dev_path = None
    # path to test corpus (only evaluate)
    test_path = 'test.tsv'
    # where to serialize evaluation result (only evaluate)
    eval_path = None
    # where to save the confusion matrix (only evaluate)
    cm_path = None


@ex.capture
def get_model_name_enum(model_name):
    return ModelName(model_name)


@ex.capture
def read_corpus(path, _log, _run, name='train', encoding='utf-8', lower=True, replace_digits=True):
    _log.info(f'Reading {name} corpus from %s', path)
    reader = CorpusReader(path, encoding=encoding, lower=lower, replace_digits=replace_digits)
    _run.add_resource(path)
    return reader


@ex.capture
def train_majority(train_path, model_path, _log, _run):
    train_reader = read_corpus(train_path)
    c = Counter(tag for _, tag in train_reader.tagged_words())
    majority_tag = c.most_common(n=1)[0][0]
    _log.info('Saving model to %s', model_path)
    with open(model_path, 'wb') as f:
        pickle.dump({'majority_tag': majority_tag}, f)
    _run.add_artifact(model_path)


@ex.capture
def train_memo(train_path, model_path, _log, _run, window=2):
    train_reader = read_corpus(train_path)
    _log.info('Start training model')
    model = MemorizationTagger.train(train_reader.tagged_sents(), window=window)
    _log.info('Saving model to %s', model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    _run.add_artifact(model_path)


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
def train_crf(train_path, model_path, _log, _run, dev_path=None):
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
    holdout = -1 if dev_path is None else 1
    trainer.train(model_path, holdout=holdout)
    _run.add_artifact(model_path)


FIELDS_FILENAME = 'fields.pkl'


@ex.capture
def save_fields(fields, save_dir, _log, _run):
    filename = os.path.join(save_dir, FIELDS_FILENAME)
    _log.info('Saving fields to %s', filename)
    torch.save(fields, filename, pickle_module=dill)
    _run.add_artifact(filename)


@ex.capture
def load_fields(save_dir, _log, _run):
    filename = os.path.join(save_dir, FIELDS_FILENAME)
    _log.info('Loading fields from %s', filename)
    fields = torch.load(filename, map_location='cpu', pickle_module=dill)
    for name, field in fields:
        assert not field.use_vocab or hasattr(field, 'vocab'), f'no vocab found for field {name}'
    _run.add_resource(filename)
    return fields


@ex.capture
def make_dataset(path, fields, _log, name='train'):
    assert len(fields) in (2, 3), 'fields should contain 2 or 3 elements'

    _log.info('Creating %s dataset', name)
    if isinstance(path, str):
        reader = read_corpus(path, name=name)
    else:
        reader = path
    examples = []
    for id_, tagged_sent in enumerate(reader.tagged_sents()):
        words, tags = zip(*tagged_sent)
        if len(fields) == 3:
            example = Example.fromlist([words, tags, id_], fields)
        else:
            example = Example.fromlist([words, tags], fields)
        examples.append(example)
    return Dataset(examples, fields)


@ex.capture
def load_fasttext_embedding(_log):
    _log.info('Loading fasttext pretrained embedding')
    ft = FastText(language='id', cache=os.path.join(os.getenv('HOME'), '.vectors_cache'))
    _log.info('Read %d pretrained words with embedding size of %d', len(ft.itos), ft.dim)
    return ft


@ex.capture
def build_vocab(fields, train_dataset, _log, min_word_freq=2, use_fasttext=False):
    WORDS, TAGS = fields[0][1], fields[1][1]

    _log.info('Building vocabulary')
    vectors = load_fasttext_embedding() if use_fasttext else None
    WORDS.build_vocab(train_dataset, min_freq=min_word_freq, vectors=vectors)
    _log.info('Found %d words', len(WORDS.vocab))
    TAGS.build_vocab(train_dataset)
    _log.info('Found %d tags', len(TAGS.vocab))


MODEL_METADATA_FILENAME = 'model_metadata.json'


@ex.capture
def save_model_metadata(metadata, save_dir, _log, _run):
    args, kwargs = metadata
    filename = os.path.join(save_dir, MODEL_METADATA_FILENAME)
    _log.info('Saving model metadata to %s', filename)
    with open(filename, 'w') as f:
        json.dump({'args': args, 'kwargs': kwargs}, f, sort_keys=True, indent=2)
    _run.add_artifact(filename)


@ex.capture
def load_model_metadata(save_dir, _log, _run):
    filename = os.path.join(save_dir, MODEL_METADATA_FILENAME)
    _log.info('Loading model metadata from %s', filename)
    with open(filename) as f:
        metadata = json.load(f)
    _run.add_resource(filename)
    return metadata


@ex.capture
def get_model_metadata(fields, training=True, word_embedding_size=100, window=2,
                       hidden_size=100, dropout=0.5, use_crf=False, padding_idx=0):
    WORDS, TAGS = fields[0][1], fields[1][1]

    if training:
        num_words, num_tags = len(WORDS.vocab), len(TAGS.vocab)
        args = (num_words, num_tags)
        kwargs = {
            'word_embedding_size': word_embedding_size,
            'window': window,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'use_crf': use_crf,
            'padding_idx': padding_idx,
            'pretrained_embedding': WORDS.vocab.vectors,
        }
        return args, kwargs

    metadata = load_model_metadata()
    args, kwargs = metadata['args'], metadata['kwargs']
    kwargs['pretrained_embedding'] = WORDS.vocab.vectors
    return args, kwargs


@ex.capture
def make_feedforward_model(fields, _log, training=True, checkpoint=None, device=-1):
    _log.info('Creating the feedforward model')
    args, kwargs = get_model_metadata(fields, training=training)
    model = FeedforwardTagger(*args, **kwargs)
    _log.info('Model created with %d parameters', sum(p.numel() for p in model.parameters()))

    if training:
        kwargs.pop('pretrained_embedding')  # this is a FloatTensor, can't save it as JSON
        save_model_metadata((args, kwargs))

    if checkpoint is not None:
        _log.info('Restoring model parameters from the checkpoint')
        model.load_state_dict(checkpoint['model'])

    if device >= 0:
        model.cuda(device)

    model.train(mode=training)
    return model


@ex.capture
def make_optimizer(model, _log, checkpoint=None, lr=0.001):
    _log.info('Creating the optimizer')
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    if checkpoint is not None:
        _log.info('Restoring optimizer parameters from the checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer


CKPT_FILENAME = 'checkpoint.pt'


@ex.capture
def save_checkpoint(state, save_dir, _log, _run, is_best=False):
    filename = os.path.join(save_dir, CKPT_FILENAME)
    _log.info('Saving checkpoint to %s', filename)
    torch.save(state, filename)
    _run.add_artifact(filename)
    if is_best:
        best_filename = os.path.join(save_dir, f'best_{CKPT_FILENAME}')
        _log.info('Copying best checkpoint to %s', best_filename)
        shutil.copyfile(filename, best_filename)
        _run.add_artifact(filename)


@ex.capture
def load_checkpoint(resume_from, _log, _run, is_best=False):
    filename = os.path.join(resume_from, f"{'best_' if is_best else ''}{CKPT_FILENAME}")
    _log.info('Loading %scheckpoint from %s', 'best ' if is_best else '', filename)
    checkpoint = torch.load(filename, map_location='cpu')
    _run.add_resource(filename)
    return checkpoint


@ex.capture
def train_feedforward(train_path, save_dir, _log, _run, dev_path=None, batch_size=16, device=-1,
                      print_every=10, max_epochs=20, stopping_patience=5, scheduler_patience=2,
                      tol=0.01, scheduler_verbose=False, resume_from=None):
    _log.info('Creating save directory %s if it does not exist', save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Create fields
    WORDS = Field(  # no `lower` because it's done in `CorpusReader`
        batch_first=True, include_lengths=True)
    TAGS = Field(batch_first=True, unk_token=None)
    fields = [('words', WORDS), ('tags', TAGS)]

    # Create datasets and iterators
    train_dataset = make_dataset(train_path, fields)
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    train_iter = BucketIterator(
        train_dataset, batch_size, sort_key=sort_key, device=device, repeat=False)
    dev_iter = None

    if dev_path is not None:
        dev_dataset = make_dataset(dev_path, fields, name='dev')
        dev_iter = BucketIterator(
            dev_dataset, batch_size, sort_key=sort_key, device=device, train=False)

    # Build vocabularies and save fields
    build_vocab(fields, train_dataset)
    save_fields(fields)

    # Create model and restore from checkpoint if given
    checkpoint = None if resume_from is None else load_checkpoint()
    model = make_feedforward_model(fields, checkpoint=checkpoint)

    # Create optimizer and learning rate scheduler
    optimizer = make_optimizer(model, checkpoint=checkpoint)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=scheduler_patience, threshold=tol,
        threshold_mode='abs', verbose=scheduler_verbose)

    # Create engine, meters, and timers
    engine = tnt.engine.Engine()
    loss_meter = tnt.meter.AverageValueMeter()
    f1_meter = tnt.meter.AverageValueMeter()
    per_tag_f1_meter = defaultdict(tnt.meter.AverageValueMeter)
    speed_meter = tnt.meter.AverageValueMeter()
    train_timer = tnt.meter.TimeMeter(None)
    epoch_timer = tnt.meter.TimeMeter(None)
    batch_timer = tnt.meter.TimeMeter(None)

    def net(minibatch):
        # shape: (batch_size, seq_length)
        words, _ = minibatch.words
        # shape: (batch_size, seq_length)
        mask = words != WORDS.vocab.stoi[WORDS.pad_token]
        # shape: (batch_size,)
        loss = model(words, minibatch.tags, mask=mask)
        # shape: (1,)
        loss = loss.mean()

        return loss, model.decode(words, mask=mask)

    def reset_meters():
        loss_meter.reset()
        f1_meter.reset()
        speed_meter.reset()
        for meter in per_tag_f1_meter.values():
            meter.reset()

    def make_checkpoint(state, is_best=False):
        save_checkpoint({
            'epoch': state['epoch'],
            't': state['t'],
            'best_f1': state['best_f1'],
            'num_bad_epochs': state['num_bad_epochs'],
            'model': model.state_dict(),
            'optimizer': state['optimizer'].state_dict(),
        }, is_best=is_best)

    def on_start(state):
        if state['train']:
            state.update({'best_f1': -float('inf'), 'num_bad_epochs': 0})
            if checkpoint is not None:
                _log.info('Resuming training from the checkpoint')
                state.update({
                    'epoch': checkpoint['epoch'],
                    't': checkpoint['t'],
                    'best_f1': checkpoint['best_f1'],
                    'num_bad_epochs': checkpoint['num_bad_epochs'],
                })
            make_checkpoint(state, is_best=True)
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
        # Compute weighted macro-averaged F1
        reference, hypothesis = [], []
        for gold, pred in zip(golds, state['output']):
            gold = gold[:len(pred)]
            reference.extend(gold.data)
            hypothesis.extend(pred)
        # Overall
        f1 = f1_score(reference, hypothesis, average='weighted')
        f1_meter.add(f1)
        # Per tag
        reference = [TAGS.vocab.itos[t] for t in reference]
        hypothesis = [TAGS.vocab.itos[t] for t in hypothesis]
        labels = list(set(reference + hypothesis))
        per_tag_f1 = f1_score(reference, hypothesis, average=None, labels=labels)
        for f1, tag in zip(per_tag_f1, labels):
            per_tag_f1_meter[tag].add(f1)

        elapsed_time = batch_timer.value()
        speed = golds.size(0) / elapsed_time
        speed_meter.add(speed)

        if state['train'] and (state['t'] + 1) % print_every == 0:
            epoch = (state['t'] + 1) / len(state['iterator'])
            batch_loss = loss_meter.value()[0]
            batch_f1 = 100 * f1_meter.value()[0]
            _log.info(
                'Epoch %.2f (%.2fms): %.2f samples/s | loss %.4f | f1 %.2f',
                epoch, 1000 * elapsed_time, speed_meter.value()[0], batch_loss, batch_f1)
            _run.log_scalar('batch_loss(train)', batch_loss, step=state['t'])
            _run.log_scalar('batch_f1(train)', batch_f1, step=state['t'])

    def on_end_epoch(state):
        elapsed_time = epoch_timer.value()
        train_loss = loss_meter.value()[0]
        train_f1 = 100 * f1_meter.value()[0]
        _log.info(
            'Epoch %d done (%.2fs): %.2f samples/s | loss %.4f | f1 %.2f',
            state['epoch'], epoch_timer.value(), speed_meter.value()[0], train_loss, train_f1)
        _run.log_scalar('loss(train)', train_loss, step=state['t'])
        _run.log_scalar('f1(train)', train_f1, step=state['t'])
        for tag, meter in per_tag_f1_meter.items():
            _run.log_scalar(f'f1(train, {tag})', meter.value()[0], step=state['t'])

        is_best = False
        if dev_iter is not None:
            _log.info('Evaluating on dev corpus')
            engine.test(net, dev_iter)
            dev_loss = loss_meter.value()[0]
            dev_f1 = 100 * f1_meter.value()[0]
            _log.info('Result on dev corpus (%.2fs): %.2f samples/s | loss %.4f | f1 %.2f',
                      epoch_timer.value(), speed_meter.value()[0], dev_loss, dev_f1)
            _run.log_scalar('loss(dev)', dev_loss, step=state['t'])
            _run.log_scalar('f1(dev)', dev_f1, step=state['t'])
            for tag, meter in per_tag_f1_meter.items():
                _run.log_scalar(f'f1(dev, {tag})', meter.value()[0], step=state['t'])

            scheduler.step(dev_f1, epoch=state['epoch'])
            if dev_f1 >= state['best_f1'] + tol:
                _log.info('New best result on dev corpus')
                state.update({'best_f1': dev_f1, 'num_bad_epochs': 0})
                is_best = True
            else:
                state['num_bad_epochs'] += 1
                if state['num_bad_epochs'] >= stopping_patience * (scheduler_patience + 1):
                    num_reduction = state['num_bad_epochs'] // (scheduler_patience + 1)
                    _log.info(
                        f"No improvements after {num_reduction} LR reductions, stopping early")
                    state['maxepoch'] = -1  # force training loop to stop

        make_checkpoint(state, is_best=is_best)

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


@ex.capture
def predict_majority(reader, model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    _run.add_resource(model_path)
    _log.info('Making predictions with the model')
    return [model['majority_tag']] * len(reader.words())


@ex.capture
def predict_memo(reader, model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    _run.add_resource(model_path)
    _log.info('Making predictions with the model')
    return [tag for sent in reader.sents() for tag in model.predict(sent)]


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
def predict_feedforward(reader, save_dir, _log, _run, device=-1, batch_size=16):
    fields = load_fields()
    WORDS, TAGS = fields[0][1], fields[1][1]
    IDS = Field(sequential=False, use_vocab=False)
    fields.append(('index', IDS))

    checkpoint = load_checkpoint(save_dir, is_best=True)
    model = make_feedforward_model(fields[:-1], training=False, checkpoint=checkpoint)

    test_dataset = make_dataset(reader, fields, name='test')
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    test_iter = BucketIterator(
        test_dataset, batch_size, sort_key=sort_key, device=device, train=False)

    indexed_predictions = []
    for minibatch in test_iter:
        # shape: (batch_size, seq_length), (batch_size,)
        words, lengths = minibatch.words
        # shape: (batch_size, seq_length)
        predictions = model.decode(words)
        # shape: (batch_size,)
        index = minibatch.index

        for i, pred, length in zip(index, predictions, lengths):
            pred = pred[:length]
            indexed_predictions.append((i.data[0], pred))

    indexed_predictions.sort()
    return [TAGS.vocab.itos[pred] for _, pred_sent in indexed_predictions for pred in pred_sent]


@ex.capture
def make_predictions(reader):
    model_name = get_model_name_enum()
    if model_name is ModelName.MAJOR:
        return predict_majority(reader)
    if model_name is ModelName.MEMO:
        return predict_memo(reader)
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
    all_labels = list(sorted(set(gold_labels + pred_labels)))
    _log.info('Saving the confusion matrix to %s', cm_path)
    cm = confusion_matrix(gold_labels, pred_labels, labels=all_labels)
    cm = cm / cm.sum(axis=1).reshape(-1, 1)
    sns.set()
    sns.heatmap(
        cm, vmin=0, vmax=1, xticklabels=all_labels, yticklabels=all_labels, cmap='YlGnBu')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(cm_path, bbox_inches='tight')
    _run.add_artifact(cm_path)


@ex.capture
def set_random_seed(_seed):
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)


@ex.command(unobserved=True)
def mongo():
    """Start an IPython/Python shell for interacting with Sacred's mongodb."""
    url = os.environ['SACRED_MONGO_URL']
    db_name = os.environ['SACRED_DB_NAME']
    client = MongoClient(url)
    db = client[db_name]
    try:
        from IPython import start_ipython
        start_ipython(argv=[], user_ns=dict(db=db))
    except ImportError:
        import code
        shell = code.InteractiveConsole(dict(db=db))
        shell.interact()


@ex.command(unobserved=True)
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


@ex.command
def evaluate(test_path, eval_path=None, cm_path=None):
    """Evaluate a trained model."""
    set_random_seed()
    reader = read_corpus(test_path, name='test')
    gold_labels = [tag for _, tag in reader.tagged_words()]
    pred_labels = make_predictions(reader)
    result = {'overall_f1': f1_score(gold_labels, pred_labels, average='weighted')}

    if eval_path is not None:
        evaluate_fully(gold_labels, pred_labels, result=result)
    if cm_path is not None:
        plot_confusion_matrix(gold_labels, pred_labels)

    return result['overall_f1']


# Commands defined after automain will not be registered. Since `train` needs to call
# `evaluate`, we need to make sure that `evaluate` is defined before `train`.
@ex.automain
def train(train_path, _log, _run, dev_path=None):
    """Train a model."""
    set_random_seed()
    if get_model_name_enum() is ModelName.MAJOR:
        train_majority()
    elif get_model_name_enum() is ModelName.MEMO:
        train_memo()
    elif get_model_name_enum() is ModelName.CRF:
        train_crf()
    else:
        train_feedforward()

    _log.info('Evaluating on train corpus')
    train_f1 = 100 * evaluate(train_path)
    _log.info('Result on train corpus: f1 %.2f', train_f1)
    _run.log_scalar('final_f1(train)', train_f1)
    if dev_path is not None:
        _log.info('Evaluating on dev corpus')
        dev_f1 = 100 * evaluate(dev_path)
        _log.info('Result on dev corpus: f1 %.2f', dev_f1)
        _run.log_scalar('final_f1(dev)', dev_f1)
