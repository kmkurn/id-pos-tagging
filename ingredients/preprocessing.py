import re

from sacred import Ingredient

ing = Ingredient('prep')


@ing.config
def cfg():
    # whether to lowercase words
    lower = True
    # whether to replace digits with zeros
    replace_digits = True


@ing.capture
def transform(w, lower=True, replace_digits=True):
    if lower:
        w = w.lower()
    if replace_digits:
        w = re.sub(r'\d+', '0', w)
    return w


def preprocess(sents):
    return [[transform(w) for w in s] for s in sents]
