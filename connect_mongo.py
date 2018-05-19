#!/usr/bin/env python

import argparse
import os

from pymongo import MongoClient


def main(args):
    pass
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Start an IPython/Python shell for interacting with Sacred's mongodb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    main(args)
