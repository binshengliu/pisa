Query an index
==============

Usage
--------------

```
queries - a tool for performing queries on an index.
Usage: ./bin/queries [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -t,--type TEXT REQUIRED     Index type
  -a,--algorithm TEXT REQUIRED
                              Query algorithm
  -i,--index TEXT REQUIRED    Collection basename
  -w,--wand TEXT              Wand data filename
  -q,--query TEXT             Queries filename
  --compressed-wand           Compressed wand input file
  -k UINT                     k value
```


Now it is possible to query the index. The command `queries` parses each line of the standard input as a tab-separated collection of term-ids, where the i-th
term is the i-th list in the input collection.

    $ ./bin/queries -t opt -a and -i test_collection.index.opt -w test_collection.wand -q ../test/test_data/queries

This performs conjunctive queries (`and`). In place of `and` other operators can
be used (`or`, `wand`, ..., see `queries.cpp`), and also multiple operators
separated by colon (`and:or:wand`).

If the WAND file is compressed, please append `--compressed-wand` flag.

Query algorithms
================

AND
--------

OR
--------

MaxScore
--------
> Howard Turtle and James Flood. 1995. Query evaluation: strategies and optimizations. Inf. Process. Manage. 31, 6 (November 1995), 831-850. DOI=http://dx.doi.org/10.1016/0306-4573(95)00020-H

WAND
----
> Andrei Z. Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient query evaluation using a two-level retrieval process. In Proceedings of the twelfth international conference on Information and knowledge management (CIKM '03). ACM, New York, NY, USA, 426-434. DOI: https://doi.org/10.1145/956863.956944

BlockMax WAND
-------------
> Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using block-max indexes. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval (SIGIR '11). ACM, New York, NY, USA, 993-1002. DOI=http://dx.doi.org/10.1145/2009916.2010048

BlockMax MaxScore
-----------------

Variable BlockMax WAND
----------------------
> Antonio Mallia, Giuseppe Ottaviano, Elia Porciani, Nicola Tonellotto, and Rossano Venturini. 2017. Faster BlockMax WAND with Variable-sized Blocks. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '17). ACM, New York, NY, USA, 625-634. DOI: https://doi.org/10.1145/3077136.3080780
