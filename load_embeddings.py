#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 7, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Lazy load for KG relation and word embeddings
'''

EMBEDDINGS_PATH = "./data/embeddings/"
KB_WORD_EMBEDDINGS_PATH = EMBEDDINGS_PATH + './data/embeddings/DBpedia_KGlove_fasttext.magnitude'
KB_RELATION_EMBEDDINGS_PATH = EMBEDDINGS_PATH + 'DBpediaVecotrs200_20Shuffle.txt'


def load_embeddings():
    kg_word_embeddings = Magnitude(KB_WORD_EMBEDDINGS_PATH)
    # kg_relation_embeddings = Magnitude(KB_RELATION_EMBEDDINGS_PATH)


if __name__ == '__main__':
    load_embeddings()
