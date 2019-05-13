from struct import unpack, iter_unpack
from itertools import zip_longest
from collections import OrderedDict
from operator import itemgetter
import numpy as np


class TermList:
    def __init__(self, index):
        fp = open(index + ".termlex", 'rb')
        count = unpack('Q', fp.read(8))[0]
        self.offsets = [x[0] for x in iter_unpack('Q', fp.read(count * 8))]
        fp.read(8)
        self.rest = fp.read()
        fp.close()

    def __iter__(self):
        for begin, end in zip_longest(self.offsets, self.offsets[1:]):
            yield self.rest[begin:end].decode('utf-8')

    def __next__(self):
        return self


class InvertedIndex:
    def __init__(self, index):
        self.docs = np.memmap(index + ".docs", dtype=np.uint32, mode='r')
        self.freqs = np.memmap(index + ".freqs", dtype=np.uint32, mode='r')

    def __iter__(self):
        i = 2
        while i < len(self.docs):
            size = self.docs[i]
            yield (self.docs[i + 1:size + i + 1],
                   self.freqs[i - 1:size + i - 1])
            i += size + 1

    def __next__(self):
        return self


class TermDocFreq:
    def __init__(self, index):
        terms = TermList(index)
        invindex = InvertedIndex(index)
        self.termdf = sorted(
            ((term, len(docs)) for term, (docs, _) in zip(terms, invindex)),
            key=itemgetter(1))

        self.termdfdict = dict(self.termdf)

    def __getitem__(self, key):
        return self.termdfdict[key]

    def nmost_frequent(self, n):
        return reversed(self.termdf[:n])

    def nleast_frequent(self, n):
        return self.termdf[n:]

    def __next__(self):
        return self


def test():
    for i, (docs, freqs) in enumerate(InvertedIndex("cw09b")):
        print(i, docs, freqs)
