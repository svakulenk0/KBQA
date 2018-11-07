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
    vectors = Magnitude(path)
    print("loaded %d vectors with %d dimensions" % (len(vectors), vectors.dim))

    # tests
    print "cat" in vectors
    print vectors.query("http://dbpedia.org/resource/Sunni_Islam")
    print vectors[42]
    print vectors.query(["http://dbpedia.org/resource/Sunni_Islam", "http://dbpedia.org/ontology/founder"])
    # print vectors.query([["http://dbpedia.org/resource/Sunni_Islam", "http://dbpedia.org/ontology/founder", "a", "book"], ["I", "read", "a", "magazine"]])
    print vectors[:42] # slice notation
    print vectors[42, 1337, 2001] # tuple notation
    print vectors.distance("http://dbpedia.org/resource/Sunni_Islam", "http://dbpedia.org/ontology/founder")
    print vectors.similarity("http://dbpedia.org/resource/Sunni_Islam", "http://dbpedia.org/ontology/founder")
    print vectors.most_similar_to_given("http://dbpedia.org/resource/Sunni_Islam", ["http://dbpedia.org/ontology/founder"])
    print vectors.most_similar("http://dbpedia.org/resource/Sunni_Islam", topn = 100) # Most similar by key
    print vectors.most_similar(vectors.query("http://dbpedia.org/resource/Sunni_Islam"), topn = 100) # Most similar by vector
    print vectors.most_similar_approx("http://dbpedia.org/resource/Sunni_Islam")
    print vectors.closer_than("http://dbpedia.org/resource/Sunni_Islam", "http://dbpedia.org/ontology/founder")

    # kg_word_embeddings_matrix = kg_word_vectors.get_vectors_mmap()
    
    vectors.close()

    # kg_relation_embeddings = Magnitude(KB_RELATION_EMBEDDINGS_PATH)


if __name__ == '__main__':
    test_load_embeddings()
