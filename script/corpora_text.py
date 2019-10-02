#!/usr/bin/env python3
import numpy as np
import sys
import os
from collections import OrderedDict
import argparse


class CorporaText:
    def __init__(self, collection_name):
        self.corpora = np.memmap(collection_name, dtype=np.uint32, mode='r')
        doc_start = self.build_start_index()

        with open(collection_name + '.documents', 'r') as f:
            docnos = f.read().split()
            self.documents = OrderedDict(zip(docnos, doc_start))

        with open(collection_name + '.terms', 'r') as f:
            self.terms = [x.rstrip('\n') for x in f]

    def doc_text(self, docno):
        if docno not in self.documents:
            return []
        start = self.documents[docno]
        size = self.corpora[start]
        seq = self.corpora[start + 1:start + 1 + size]
        return [self.terms[x] for x in seq]

    def build_start_index(self):
        i = 2
        starting = []
        while i < len(self.corpora):
            size = self.corpora[i]
            starting.append(i)
            i += size + 1
        return starting

    def __len__(self):
        return self.corpora[1]

    def __iter__(self):
        for docno in self.documents:
            yield (docno, self.doc_text(docno))

    def __next__(self):
        return self


def file_path(prefix):
    if not all(
            os.path.isfile(p)
            for p in [prefix, prefix + '.documents', prefix + '.terms']):
        raise argparse.ArgumentTypeError("Please check the prefix is valid.")
    return prefix


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""Dump original document text.
With --interactive, docnos are read from stdin.
Without --interactive, all documents are dumped.""",
        epilog="""examples:
  $ corpora_text.py ./robust04 > robust04.text
  $ echo -e 'FBIS3-10001\\nFBIS3-10006\\nFBIS3-10009' | corpora_text.py --interactive ./robust04""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i',
                        '--interactive',
                        action='store_true',
                        help='read docno from stdin')

    parser.add_argument(
        'prefix',
        metavar='forward_index_name',
        type=file_path,
        help=
        'forward index filename specified to "parse_collection -o", e.g. ./robust'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    corpora = CorporaText(args.prefix)

    if args.interactive:
        print('Type docno and press return (ctrl-d to quit):', file=sys.stderr)
        for docno in sys.stdin:
            doc_text = ' '.join(corpora.doc_text(docno.strip()))
            print('{}\t{}'.format(docno.strip(), doc_text))
        return

    total = len(corpora)
    for idx, (docno, text) in enumerate(corpora):
        doc_text = ' '.join(text)
        print('{}\t{}'.format(docno, doc_text))


if __name__ == '__main__':
    main()