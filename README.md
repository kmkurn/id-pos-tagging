# Indonesian Part-of-Speech (POS) Tagging

This repository contains the implementation of our paper:

*Toward a Standardized and More Accurate Indonesian Part-of-Speech Tagging*.
Kemal Kurniawan and Alham Fikri Aji. In *Proceedings of the International
Conference on Asian Language Processing* (to appear). 2018.

## Requirements

Make sure you have [conda package manager](https://conda.io/docs/). Then, create
a conda virtual environment with

```bash
$ conda env create -f environment.yml
```

The command will create a virtual environment named `id-pos-tagging` and also
install all the required packages. Once it is done, activate the virtual
environment to get started.

```bash
$ source activate id-pos-tagging
```

## Dataset

The dataset is available in `data/dataset.tar.gz`. Decompress this file and you
will have `train.X.tsv`, `dev.X.tsv`, and `test.X.tsv` files for all 5 folds
with `X` replaced with the fold number.

## Running experiments

Scripts to run our models are prefixed with `run_`. So, for example, to run the
CRF model, use `run_crf.py` script. All scripts use [Sacred](http://sacred.readthedocs.io/)
to manage the experiment configuration and results. We will explain in a more detail and use
this `run_crf.py` script as the example.

### Training

A minimal command to train a model is

```bash
$ ./run_crf.py train with corpus.train=train.01.tsv
```

This will train a CRF model on the given training corpus and save the model in `model` file
in the current directory. There are many configuration that can be set, which can all be
listed with

```bash
$ ./run_crf.py print_config
```

The command above will show all the configuration for the script, including those that
might be needed for commands other than `train`. The `print_config` command is available
for other `run_*.py` scripts as well.

To make reproduction easier, we already named our best configuration reported in the paper as
`tuned_on_foldX` where `X` is the fold number. For instance, to get our result on fold 1, run

```bash
$ ./run_crf.py train with tuned_on_fold1 corpus.train=train.01.tsv
```

These named configurations are also available for `run_memo.py` and `run_neural.py`.

### Evaluation and prediction

To evaluate/predict, use `evaluate` and `predict` command respectively. The available
configuration is still the same as that of training.

## Observing experiments

Sacred allows us to save experiment runs to a MongoDB database. To enable this for our scripts,
simply set `SACRED_MONGO_URL` and `SACRED_DB_NAME` to your MongoDB instance. Once this is done,
an experiment run will be saved everytime you run any `run_*.py` scripts.

## License

MIT
