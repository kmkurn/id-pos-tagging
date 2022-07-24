#!/usr/bin/env python

import argparse
import glob
import os
import random

from utils import CorpusReader


def main(args):
    random.seed(args.seed)
    reader = CorpusReader(args.corpus_path, encoding=args.encoding)
    tagged_sents = list(reader.tagged_sents())
    random.shuffle(tagged_sents)
    id2taggedsents = dict(enumerate(tagged_sents))

    for name in ('train', 'dev', 'test'):
        for path in glob.glob(os.path.join(args.splits_dir, f'{name}.*.txt')):
            tagged_sents = []
            with open(path, encoding=args.encoding) as f:
                for line in f:
                    id_ = int(line.strip())
                    tagged_sents.append(id2taggedsents[id_])
            opath = os.path.splitext(path)[0] + '.tsv'
            write_tsv(tagged_sents, opath, args.encoding)


def write_tsv(tagged_sents, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as f:
        for tagged_sent in tagged_sents:
            print('\n'.join('\t'.join(pair) for pair in tagged_sent), file=f, end='\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create splits from a TSV corpus.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('splits_dir', help='directory of the split files')
    parser.add_argument('corpus_path', help='path to the TSV corpus file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument('--seed', type=int, default=12345, help='random seed used to generate the splits')
    args = parser.parse_args()
    main(args)
