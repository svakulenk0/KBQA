#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 7, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Lazy load for KG relation and word embeddings
'''
from pymagnitude import *


EMBEDDINGS_PATH = "./data/embeddings/"
KB_WORD_EMBEDDINGS_PATH = EMBEDDINGS_PATH + 'DBpedia_KGlove_fasttext.magnitude'
# KB_RELATION_EMBEDDINGS_PATH = EMBEDDINGS_PATH + 'DBpediaVecotrs200_20Shuffle.txt'


def test_load_embeddings(path=KB_WORD_EMBEDDINGS_PATH):
    vectors = Magnitude(path)  # very fast
    print("loaded %d vectors with %d dimensions" % (len(vectors), vectors.dim))  # very fast

    # tests
    print "Sunni_Islam" in vectors  # very fast

    # get vectors
        # print vectors.query("Sunni_Islam")  # very fast
        # print vectors[42]  # very fast
        # print vectors.query(["Sunni_Islam", "founder"])  # very fast
        # # print vectors.query([["Sunni_Islam", "founder", "a", "book"], ["I", "read", "a", "magazine"]])
        # print vectors[:42] # slice notation
        # print vectors[42, 1337, 2001] # tuple notation

        # # pairwise comparison
        # print vectors.distance("Sunni_Islam", "founder")  # very fast
        # print vectors.similarity("Sunni_Islam", "founder")  # very fast
        # print vectors.most_similar_to_given("Sunni_Islam", ["founder"])  # very fast

    # request KB
    print vectors.most_similar("Sunni_Islam", topn = 10) # Most similar by key
    print vectors.most_similar(vectors.query("Sunni_Islam"), topn = 10) # Most similar by vector
    print vectors.most_similar_approx("Sunni_Islam")
    
    print vectors.closer_than("Sunni_Islam", "founder")  # throws some error

    # kg_word_embeddings_matrix = kg_word_vectors.get_vectors_mmap()  # takes some time but possible
    
    vectors.close()

    # kg_relation_embeddings = Magnitude(KB_RELATION_EMBEDDINGS_PATH)


if __name__ == '__main__':
    test_load_embeddings()
