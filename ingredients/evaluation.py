import json

from sacred import Ingredient
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ingredients.corpus import ing as corpus_ingredient, read_train_corpus, read_dev_corpus, \
    read_test_corpus
from utils import SACRED_OBSERVE_FILES

ing = Ingredient('eval', ingredients=[corpus_ingredient])


@ing.config
def cfg():
    # which set of the corpus to evaluate on [train, dev, test]
    which = 'test'
    # where to serialize the full evaluation result
    path = None
    # where to save the confusion matrix
    cm_path = None


@ing.capture
def evaluate_fully(gold_labels, pred_labels, path, _log, _run, result=None):
    if result is None:
        result = {}

    all_labels = list(set(gold_labels + pred_labels))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=all_labels)
    for label, p, r, f in zip(all_labels, prec, rec, f1):
        result[label] = {'P': p, 'R': r, 'F1': f}
    _log.info('Saving the full evaluation result to %s', path)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(path)


@ing.capture
def plot_confusion_matrix(gold_labels, pred_labels, cm_path, _log, _run):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        # maybe on a machine without display?
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

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
    if SACRED_OBSERVE_FILES:
        _run.add_artifact(cm_path)


@ing.capture
def run_evaluation(make_predictions, _log, _run, which='test', path=None, cm_path=None):
    read_fn = {
        'train': read_train_corpus,
        'dev': read_dev_corpus,
        'test': read_test_corpus,
    }

    try:
        reader = read_fn[which]()
    except KeyError:
        choices = ', '.join(read_fn.keys())
        msg = f'{which} is not a valid corpus set, possible choices are: {choices}'
        raise KeyError(msg)

    _log.info('Obtaining the gold labels from the %s corpus', which)
    gold_labels = [tag for _, tag in reader.tagged_words()]
    _log.info('Obtaining the predicted labels')
    pred_labels = make_predictions(reader.sents())
    f1 = f1_score(gold_labels, pred_labels, average='weighted')
    _log.info('f1: %f', f1)
    _run.log_scalar('f1', f1)

    if path is not None:
        evaluate_fully(gold_labels, pred_labels, result={'overall_f1': f1})

    if cm_path is not None:
        plot_confusion_matrix(gold_labels, pred_labels)

    return f1
