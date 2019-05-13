from inverted_index import TermDocFreq
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dump term doc-frequency.')
    parser.add_argument('index')
    return parser.parse_args()


def main():
    args = parse_arguments()
    term_freq = TermDocFreq(args.index)
    for term, freq in term_freq.nmost_frequent(None):
        print('{} {}'.format(term, freq))


if __name__ == '__main__':
    main()
