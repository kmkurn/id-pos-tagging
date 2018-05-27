#!/usr/bin/env python

import collections
import dill
import enum
import math
import operator
import os
import shutil

from sacred import Experiment
from sklearn.metrics import f1_score
from torchtext.data import BucketIterator, Dataset, Example, Field, NestedField
from torchtext.vocab import FastText
import torch
import torch.optim as optim
import torchnet as tnt

from ingredients.corpus import ing as corpus_ingredient, read_train_corpus, read_dev_corpus
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.preprocessing import ing as prep_ingredient, preprocess
from models.tagger import make_neural_tagger
from serialization import dump, load
from utils import SACRED_OBSERVE_FILES, run_predict, separate_tagged_sents, setup_mongo_observer

ingredients = [corpus_ingredient, eval_ingredient, prep_ingredient]
ex = Experiment(name='id-pos-tagging-neural-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


class Comparing(enum.Enum):
    F1 = 'f1'
    LOSS = 'loss'


@ex.config
def default():
    # save models and checkpoints here
    save_dir = 'save_dir'
    # whether to overwrite save_dir
    overwrite = False
    # resume training from the checkpoint saved in this directory
    resume_from = None
    # whether to lowercase words
    lower_words = True
    # exclude words occurring fewer than this count from vocab
    min_word_freq = 2
    # size of word embedding
    word_embedding_size = 100
    # whether to use prefix features
    use_prefix = False
    # whether to lowercase prefixes
    lower_prefixes = True
    # exclude prefixes occurring fewer than this count from vocab
    min_prefix_freq = 5
    # size of prefix embedding, can be a 2-tuple for 2- and 3-prefix
    prefix_embedding_size = 20
    # whether to use suffix features
    use_suffix = False
    # whether to lowercase suffixes
    lower_suffixes = True
    # exclude suffixes occurring fewer than this count from vocab
    min_suffix_freq = 5
    # size of suffix embedding, can be a 2-tuple for 2- and 3-suffix
    suffix_embedding_size = 20
    # whether to use character features
    use_chars = False
    # whether to lowercase chars
    lower_chars = False
    # size of character embedding
    char_embedding_size = 30
    # number of character convolution filters
    num_char_filters = 30
    # width of each filter
    filter_width = 3
    # size of hidden layer
    hidden_size = 100
    # dropout rate
    dropout = 0.5
    # whether to apply a biLSTM layer after embedding layers
    use_lstm = False
    # context window size (defaults to 0 if use_lstm=True)
    window = 2 if not use_lstm else 0
    # whether to use CRF layer as output layer instead of softmax
    use_crf = False
    # learning rate
    lr = 0.001
    # batch size at train time
    batch_size = 8
    # batch size at test time
    test_batch_size = 256
    # GPU device or -1 for CPU
    device = 0 if torch.cuda.is_available() else -1
    # print training log every this iterations
    print_every = 10
    # maximum number of training epochs
    max_epochs = 50
    # what dev score to compare on end epoch [f1, loss]
    comparing = Comparing.F1.value
    # wait for this number of LR reduction before early stopping
    stopping_patience = 5
    # reduce LR when no new best score in this number of epochs
    scheduler_patience = 2
    # tolerance for comparing score for early stopping
    tol = 1e-4
    # whether to print to stdout when LR is reduced
    scheduler_verbose = False
    # use fasttext pretrained word embedding
    use_fasttext = False
    # normalize gradient at this threshold
    grad_norm_threshold = 1.


# Disable lowercasing in preprocessing by the preprocessing ingredient
# because the neural model may lowercase the words but not the subwords
@prep_ingredient.config
def update_cfg():
    lower = False


FIELDS_FILENAME = 'fields.pkl'


@ex.capture
def save_fields(field_odict, save_dir, _log, _run):
    filename = os.path.join(save_dir, FIELDS_FILENAME)
    _log.info('Saving fields to %s', filename)
    torch.save(field_odict, filename, pickle_module=dill)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(filename)


@ex.capture
def load_fields(save_dir, _log, _run):
    filename = os.path.join(save_dir, FIELDS_FILENAME)
    _log.info('Loading fields from %s', filename)
    field_odict = torch.load(filename, map_location='cpu', pickle_module=dill)
    for name, field in field_odict.items():
        assert field is None or not field.use_vocab or hasattr(
            field, 'vocab'), f'no vocab found for field {name}'
    if SACRED_OBSERVE_FILES:
        _run.add_resource(filename)
    return field_odict


CKPT_FILENAME = 'checkpoint.pt'


@ex.capture
def save_checkpoint(state, save_dir, _log, _run, is_best=False):
    filename = os.path.join(save_dir, CKPT_FILENAME)
    _log.info('Saving checkpoint to %s', filename)
    torch.save(state, filename)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(filename)
    if is_best:
        best_filename = os.path.join(save_dir, f'best_{CKPT_FILENAME}')
        _log.info('Copying best checkpoint to %s', best_filename)
        shutil.copyfile(filename, best_filename)
        if SACRED_OBSERVE_FILES:
            _run.add_artifact(filename)


@ex.capture
def load_checkpoint(resume_from, _log, _run, is_best=False):
    filename = os.path.join(resume_from, f"{'best_' if is_best else ''}{CKPT_FILENAME}")
    _log.info('Loading %scheckpoint from %s', 'best ' if is_best else '', filename)
    checkpoint = torch.load(filename, map_location='cpu')
    if SACRED_OBSERVE_FILES:
        _run.add_resource(filename)
    return checkpoint


METADATA_FILENAME = 'metadata'


@ex.capture
def save_metadata(metadata, save_dir, _log, _run):
    filename = os.path.join(save_dir, METADATA_FILENAME)
    _log.info('Saving metadata to %s', filename)
    with open(filename, 'w') as f:
        print(dump(metadata), file=f)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(filename)


@ex.capture
def load_metadata(save_dir, _log, _run):
    filename = os.path.join(save_dir, METADATA_FILENAME)
    _log.info('Loading metadata from %s', filename)
    with open(filename) as f:
        metadata = load(f.read())
    if SACRED_OBSERVE_FILES:
        _run.add_resource(filename)
    return metadata


@ex.capture
def get_metadata(
        field_odict,
        use_prefix=False,
        use_suffix=False,
        use_chars=False,
        word_embedding_size=100,
        prefix_embedding_size=20,
        suffix_embedding_size=20,
        char_embedding_size=30,
        num_char_filters=100,
        filter_width=3,
        window=2,
        hidden_size=100,
        dropout=0.5,
        use_lstm=False,
        use_crf=False):
    WORDS, TAGS = field_odict['words'], field_odict['tags']

    metadata = {
        'num_words': len(WORDS.vocab),
        'num_tags': len(TAGS.vocab),
        'num_prefixes': None,
        'num_suffixes': None,
        'num_chars': None,
        'word_embedding_size': word_embedding_size,
        'prefix_embedding_size': prefix_embedding_size,
        'suffix_embedding_size': suffix_embedding_size,
        'char_embedding_size': char_embedding_size,
        'num_char_filters': num_char_filters,
        'filter_width': filter_width,
        'window': window,
        'hidden_size': hidden_size,
        'dropout': dropout,
        'use_lstm': use_lstm,
        'use_crf': use_crf,
        'padding_idx': WORDS.vocab.stoi[WORDS.pad_token],
    }
    if use_prefix:
        PREFIXES_2, PREFIXES_3 = field_odict['prefs_2'], field_odict['prefs_3']
        metadata['num_prefixes'] = (len(PREFIXES_2.vocab), len(PREFIXES_3.vocab))
    if use_suffix:
        SUFFIXES_2, SUFFIXES_3 = field_odict['suffs_2'], field_odict['suffs_3']
        metadata['num_suffixes'] = (len(SUFFIXES_2.vocab), len(SUFFIXES_3.vocab))
    if use_chars:
        CHARS = field_odict['chars']
        metadata['num_chars'] = len(CHARS.vocab)
    return metadata


@ex.capture
def make_model(metadata, _log, checkpoint=None, device=-1, pretrained_embedding=None):
    _log.info('Creating the neural model')
    model = make_neural_tagger(pretrained_embedding=pretrained_embedding, **metadata)
    _log.info('Model created with %d parameters', sum(p.numel() for p in model.parameters()))

    if checkpoint is not None:
        _log.info('Restoring model parameters from the checkpoint')
        model.load_state_dict(checkpoint['model'])

    if device >= 0:
        model.cuda(device)

    return model


@ex.capture
def load_model(field_odict, save_dir):
    metadata = load_metadata()
    WORDS = field_odict['words']
    checkpoint = load_checkpoint(save_dir, is_best=True)
    model = make_model(
        metadata, checkpoint=checkpoint, pretrained_embedding=WORDS.vocab.vectors)
    return model


@ex.capture
def make_dataset(sents, fields, _log, tags=None):
    assert sents, 'no sentences found'
    assert tags is None or len(sents) == len(tags)

    if tags is None:
        tags = [None] * len(sents)

    _log.info('Creating dataset')

    examples = []
    for id_, (words, tags_) in enumerate(zip(sents, tags)):
        prefs_2 = [w[:2] for w in words]
        prefs_3 = [w[:3] for w in words]
        suffs_2 = [w[-2:] for w in words]
        suffs_3 = [w[-3:] for w in words]
        data = [words, tags_, prefs_2, prefs_3, suffs_2, suffs_3, words, id_]
        examples.append(Example.fromlist(data, fields))

    return Dataset(examples, fields)


@ex.capture
def make_preds(field_odict, model, sents, _log, test_batch_size=32, device=-1):
    TAGS = field_odict['tags']

    # We need to set tags field to None because we don't have tags at test time; setting it
    # to None skips it when creating examples
    field_odict = collections.OrderedDict(field_odict)  # shallow copy
    field_odict['tags'] = None

    # We need index field to sort the examples back to its original order because
    # the iterator will sort them according to length to minimize padding
    field_odict['index'] = Field(sequential=False, use_vocab=False)

    dataset = make_dataset(preprocess(sents), field_odict.items())
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    iterator = BucketIterator(
        dataset, test_batch_size, sort_key=sort_key, device=device, train=False)

    _log.info('Making predictions with the model')

    ix_preds = []
    for minibatch in iterator:
        # shape: (batch_size,)
        ix = minibatch.index
        # shape: (batch_size, seq_length)
        inputs = [minibatch.words]

        if hasattr(minibatch, 'prefs_2'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.prefs_2)

        if hasattr(minibatch, 'prefs_3'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.prefs_3)

        if hasattr(minibatch, 'suffs_2'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.suffs_2)

        if hasattr(minibatch, 'suffs_3'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.suffs_3)

        if hasattr(minibatch, 'chars'):
            # shape: (batch_size, seq_length, num_chars)
            inputs.append(minibatch.chars)

        preds = model.decode(inputs)
        for i, pred, in zip(ix, preds):
            pred = pred[1:-1]  # strip init and eos tokens
            ix_preds.append((int(i), pred))

    ix_preds.sort()
    return [TAGS.vocab.itos[p] for _, ps in ix_preds for p in ps]


@ex.capture
def create_fields(
        use_prefix=False,
        use_suffix=False,
        use_chars=False,
        lower_words=True,
        lower_prefixes=True,
        lower_suffixes=True,
        lower_chars=False):
    WORDS = Field(batch_first=True, lower=lower_words, init_token='<s>', eos_token='</s>')
    TAGS = Field(batch_first=True, init_token='<s>', eos_token='</s>')
    PREFIXES_2 = Field(
        batch_first=True, lower=lower_prefixes, init_token='<s>', eos_token='</s>')
    PREFIXES_3 = Field(
        batch_first=True, lower=lower_prefixes, init_token='<s>', eos_token='</s>')
    SUFFIXES_2 = Field(
        batch_first=True, lower=lower_suffixes, init_token='<s>', eos_token='</s>')
    SUFFIXES_3 = Field(
        batch_first=True, lower=lower_suffixes, init_token='<s>', eos_token='</s>')
    CHARS = NestedField(
        Field(
            batch_first=True,
            lower=lower_chars,
            pad_token='<cpad>',
            unk_token='<cunk>',
            tokenize=list,
            init_token='<w>',
            eos_token='</w>'),
        init_token='<s>',
        eos_token='</s>')

    field_odict = collections.OrderedDict({
        'words': WORDS,
        'tags': TAGS,
        'prefs_2': None,
        'prefs_3': None,
        'suffs_2': None,
        'suffs_3': None,
        'chars': None,
    })
    if use_prefix:
        field_odict['prefs_2'] = PREFIXES_2
        field_odict['prefs_3'] = PREFIXES_3
    if use_suffix:
        field_odict['suffs_2'] = SUFFIXES_2
        field_odict['suffs_3'] = SUFFIXES_3
    if use_chars:
        field_odict['chars'] = CHARS

    return field_odict


@ex.capture
def load_fasttext(_log):
    _log.info('Loading fasttext pretrained embedding')
    ft = FastText(language='id', cache=os.path.join(os.getenv('HOME'), '.vectors_cache'))
    _log.info('Read %d pretrained words with embedding size of %d', len(ft.itos), ft.dim)
    return ft


@ex.capture
def build_vocab(
        fields,
        dataset,
        _log,
        min_word_freq=2,
        use_fasttext=False,
        min_prefix_freq=5,
        min_suffix_freq=5):
    assert fields, 'fields should not be empty'

    _log.info('Building vocabulary')

    vectors = load_fasttext() if use_fasttext else None
    for name, field in fields:
        if field is not None:
            kwargs = {}
            if name == 'words':
                kwargs['min_freq'] = min_word_freq
                kwargs['vectors'] = vectors
            elif name.startswith('prefs'):
                kwargs['min_freq'] = min_prefix_freq
            elif name.startswith('suffs'):
                kwargs['min_freq'] = min_suffix_freq
            field.build_vocab(dataset, **kwargs)
            _log.info('Found %d %s', len(field.vocab), name)


@ex.capture
def make_optimizer(model, _log, checkpoint=None, lr=0.001, device=-1):
    _log.info('Creating the optimizer')
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    if checkpoint is not None:
        _log.info('Restoring optimizer parameters from the checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer'])

    if device >= 0:
        # move optimizer states to CUDA if necessary
        # see https://github.com/pytorch/pytorch/issues/2830
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)

    return optimizer


@ex.capture
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@ex.command
def train(
        save_dir,
        _log,
        _run,
        batch_size=16,
        test_batch_size=32,
        device=-1,
        print_every=10,
        max_epochs=20,
        stopping_patience=5,
        scheduler_patience=2,
        tol=0.01,
        scheduler_verbose=False,
        resume_from=None,
        overwrite=False,
        comparing=Comparing.F1.value,
        grad_norm_threshold=1.):
    """Train a neural tagger."""
    set_random_seed()
    _log.info('Creating save directory %s if it does not exist', save_dir)
    os.makedirs(save_dir, exist_ok=overwrite)

    # Create fields
    field_odict = create_fields()
    WORDS = field_odict['words']

    # Create datasets and iterators
    reader = read_train_corpus()
    sents, tags = separate_tagged_sents(reader.tagged_sents())
    train_dataset = make_dataset(preprocess(sents), field_odict.items(), tags=tags)
    sort_key = lambda ex: len(ex.words)  # noqa: E731
    train_iter = BucketIterator(
        train_dataset, batch_size, sort_key=sort_key, device=device, repeat=False)
    train_eval_iter = BucketIterator(
        train_dataset, test_batch_size, sort_key=sort_key, device=device, train=False)
    dev_iter = None

    reader = read_dev_corpus()
    if reader is not None:
        sents, tags = separate_tagged_sents(reader.tagged_sents())
        dev_dataset = make_dataset(preprocess(sents), field_odict.items(), tags=tags)
        dev_iter = BucketIterator(
            dev_dataset, test_batch_size, sort_key=sort_key, device=device, train=False)

    # Build vocabularies and save fields
    build_vocab(field_odict.items(), train_dataset)
    save_fields(field_odict)

    # Create model and restore from checkpoint if given
    metadata = get_metadata(field_odict)
    checkpoint = None if resume_from is None else load_checkpoint()
    model = make_model(
        metadata, checkpoint=checkpoint, pretrained_embedding=WORDS.vocab.vectors)
    model.train()
    save_metadata(metadata)

    # Create optimizer and learning rate scheduler
    optimizer = make_optimizer(model, checkpoint=checkpoint)
    comp = Comparing(comparing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max' if comp is Comparing.F1 else 'min',
        factor=0.5,
        patience=scheduler_patience,
        threshold=tol,
        threshold_mode='abs',
        verbose=scheduler_verbose)

    # Create engine, meters, timers, etc
    engine = tnt.engine.Engine()
    loss_meter = tnt.meter.AverageValueMeter()
    speed_meter = tnt.meter.AverageValueMeter()
    references = []
    hypotheses = []
    train_timer = tnt.meter.TimeMeter(None)
    epoch_timer = tnt.meter.TimeMeter(None)
    batch_timer = tnt.meter.TimeMeter(None)
    comp_op = operator.ge if comp is Comparing.F1 else operator.le
    sign = 1 if comp is Comparing.F1 else -1

    def net(minibatch):
        # shape: (batch_size, seq_length)
        inputs = [minibatch.words]

        if hasattr(minibatch, 'prefs_2'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.prefs_2)

        if hasattr(minibatch, 'prefs_3'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.prefs_3)

        if hasattr(minibatch, 'suffs_2'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.suffs_2)

        if hasattr(minibatch, 'suffs_3'):
            # shape: (batch_size, seq_length)
            inputs.append(minibatch.suffs_3)

        if hasattr(minibatch, 'chars'):
            # shape: (batch_size, seq_length, num_chars)
            inputs.append(minibatch.chars)

        # shape: (1,)
        loss = model(inputs, minibatch.tags)

        return loss, model.decode(inputs)

    def reset_meters():
        loss_meter.reset()
        speed_meter.reset()
        nonlocal references, hypotheses
        references = []
        hypotheses = []

    def make_checkpoint(state, is_best=False):
        save_checkpoint({
            'epoch': state['epoch'],
            't': state['t'],
            'best_score': state['best_score'],
            'num_bad_epochs': state['num_bad_epochs'],
            'model': model.state_dict(),
            'optimizer': state['optimizer'].state_dict(),
        }, is_best=is_best)  # yapf: disable

    def evaluate_on(name):
        assert name in ('train', 'dev')

        iterator = train_eval_iter if name == 'train' else dev_iter
        _log.info('Evaluating on %s', name)
        engine.test(net, iterator)
        loss = loss_meter.mean
        f1 = f1_score(references, hypotheses, average='weighted')
        _log.info(
            '** Result on %s (%.2fs): %.2f samples/s | loss %.4f | ppl %.4f | f1 %s',
            name.upper(), epoch_timer.value(), speed_meter.mean, loss, math.exp(loss),
            f'{f1:.2%}')
        _run.log_scalar(f'loss({name})', loss)
        _run.log_scalar(f'ppl({name})', math.exp(loss))
        _run.log_scalar(f'f1({name})', f1)

        return loss, f1

    def on_start(state):
        if state['train']:
            state.update({
                'best_score': -sign * float('inf'),
                'num_bad_epochs': 0,
            })

            if checkpoint is not None:
                _log.info('Resuming training from the checkpoint')
                state.update({
                    'epoch': checkpoint['epoch'],
                    't': checkpoint['t'],
                    'best_score': checkpoint['best_score'],
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
        if state['train']:
            torch.nn.utils.clip_grad_norm((p for p in model.parameters() if p.requires_grad),
                                          grad_norm_threshold)

        batch_loss = float(state['loss'])
        loss_meter.add(batch_loss)
        # shape: (batch_size, seq_length)
        golds = state['sample'].tags

        elapsed_time = batch_timer.value()
        batch_speed = golds.size(0) / elapsed_time
        speed_meter.add(batch_speed)

        if not state['train']:
            for gold, pred in zip(golds, state['output']):
                gold = gold.data[:len(pred)]
                assert gold[0] == WORDS.vocab.stoi[WORDS.init_token]
                assert gold[-1] == WORDS.vocab.stoi[WORDS.eos_token]
                gold, pred = gold[1:-1], pred[1:-1]  # strip init and eos tokens
                references.extend(gold)
                hypotheses.extend(pred)
        elif (state['t'] + 1) % print_every == 0:
            batch_ref, batch_hyp = [], []
            for gold, pred in zip(golds, state['output']):
                gold = gold.data[:len(pred)]
                assert gold[0] == WORDS.vocab.stoi[WORDS.init_token]
                assert gold[-1] == WORDS.vocab.stoi[WORDS.eos_token]
                gold, pred = gold[1:-1], pred[1:-1]  # strip init and eos tokens
                batch_ref.extend(gold)
                batch_hyp.extend(pred)

            batch_f1 = f1_score(batch_ref, batch_hyp, average='weighted')
            epoch = (state['t'] + 1) / len(state['iterator'])
            _log.info(
                'Epoch %.2f (%5.2fms): %.2f samples/s | loss %.4f | ppl %.4f | f1 %s', epoch,
                1000 * elapsed_time, batch_speed, batch_loss, math.exp(batch_loss),
                f'{batch_f1:.2%}')
            _run.log_scalar('batch_loss(train)', batch_loss, step=state['t'])
            _run.log_scalar('batch_ppl(train)', math.exp(batch_loss), step=state['t'])
            _run.log_scalar('batch_f1(train)', batch_f1, step=state['t'])

    def on_end_epoch(state):
        _log.info(
            'Epoch %d done (%.2fs): mean speed %.2f samples/s | mean loss %.4f | mean ppl %.4f',
            state['epoch'], epoch_timer.value(), speed_meter.mean, loss_meter.mean,
            math.exp(loss_meter.mean))
        evaluate_on('train')

        is_best = False
        if dev_iter is not None:
            dev_loss, dev_f1 = evaluate_on('dev')
            score = dev_f1 if comp is Comparing.F1 else dev_loss
            scheduler.step(score, epoch=state['epoch'])

            if comp_op(score, state['best_score'] + sign * tol):
                _log.info('** NEW best result on dev corpus')
                state.update({'best_score': score, 'num_bad_epochs': 0})
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


@ex.command(unobserved=True)
def predict():
    """Make predictions using a trained neural model."""
    field_odict = load_fields()
    model = load_model(field_odict)
    model.eval()
    run_predict(lambda sents: make_preds(field_odict, model, sents))


@ex.automain
def evaluate():
    """Evaluate a trained neural tagger."""
    field_odict = load_fields()
    model = load_model(field_odict)
    model.eval()
    return run_evaluation(lambda sents: make_preds(field_odict, model, sents))
