#!/usr/bin/env python

import argparse
import math
import os
import random

from utils import CorpusReader


def main(args):
    if args.num_folds <= 1:
        raise ValueError('number of folds is at least 2, found', args.num_folds)

    random.seed(args.seed)
    reader = CorpusReader(args.path, encoding=args.encoding)
    tagged_sents = list(reader.tagged_sents())
    random.shuffle(tagged_sents)

    for fold in range(args.num_folds):
        test, rest = [], []
        for id_, _ in enumerate(tagged_sents):
            if id_ % args.num_folds == fold:
                test.append(id_)
            else:
                rest.append(id_)
        n_dev = math.floor(args.dev * len(rest))
        dev, train = rest[:n_dev], rest[n_dev:]

        for split, name in ((train, 'train'), (dev, 'dev'), (test, 'test')):
            filename = f'{name}.{fold+1:02}.txt'
            path = os.path.join(args.outdir, filename)
            with open(path, 'w', encoding=args.encoding) as f:
                for id_ in split:
                    print(id_, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create splits from a TSV corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path to the TSV corpus file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument(
        '--dev', type=float, default=0.1, help='proportion of dev set (from train set)')
    parser.add_argument('--num-folds', '-k', type=int, default=5, help='number of folds')
    parser.add_argument(
        '-o', '--output-directory', dest='outdir', default=os.getcwd(), help='output directory')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    args = parser.parse_args()
    main(args)
