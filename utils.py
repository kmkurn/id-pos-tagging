import os

from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import BlanklineTokenizer, RegexpTokenizer


class CorpusReader(TaggedCorpusReader):
    def __init__(self, path: str, encoding: str = 'utf8'):
        word_tokenizer = RegexpTokenizer(r'\n', gaps=True)
        sent_tokenizer = BlanklineTokenizer()

        def para_block_reader(stream):
            return [stream.read()]

        super().__init__(
            *os.path.split(path), sep='\t', word_tokenizer=word_tokenizer,
            sent_tokenizer=sent_tokenizer, para_block_reader=para_block_reader,
            encoding=encoding)
