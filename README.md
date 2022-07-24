# Indonesian Part-of-Speech (POS) Tagging

This repository contains the implementation of our paper:

Kurniawan, K., & Aji, A. F. (2018). Toward a Standardized and
More Accurate Indonesian Part-of-Speech Tagging. 2018 International
Conference on Asian Language Processing (IALP), 303–307.
https://doi.org/10.1109/IALP.2018.8629236

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
will have `train.X.txt`, `dev.X.txt`, and `test.X.txt` files for all 5 folds
with `X` replaced with the fold number. Each file contains the indices of the sentences
in the original corpus. To obtain the sentences, you must first download the
[IDN Tagged Corpus](https://github.com/famrashel/idn-tagged-corpus). Then, run

```bash
$ ./splits2tsv.py data Indonesian_Manually_Tagged_Corpus.tsv
```

where `data` is the directory containing the `{train,dev,test}.X.txt` files. The
sentences will then be available in `data/{train,dev,test}.X.tsv` files.

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

## Citation

If you use our work, please cite:

```
@inproceedings{kurniawan2018,
  place={Bandung, Indonesia},
  title={Toward a Standardized and More Accurate Indonesian Part-of-Speech Tagging},
  url={https://ieeexplore.ieee.org/document/8629236},
  DOI={10.1109/IALP.2018.8629236},
  note={arXiv: 1809.03391},
  booktitle={2018 International Conference on Asian Language Processing (IALP)},
  publisher={IEEE},
  author={Kurniawan, Kemal and Aji, Alham Fikri},
  year={2018},
  month={Nov},
  pages={303–307}
}
```
