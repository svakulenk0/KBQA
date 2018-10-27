#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 27, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Entity Linking layer trained to align word and entity embeddings.

'''
from keras import backend as K


def EntityLinking(question_words_embeddings, kg_word_embeddings_matrix, kg_relation_embeddings_matrix):

        kg_word_embeddings = K.constant(kg_word_embeddings_matrix.T)
        kg_relation_embeddings = K.constant(kg_relation_embeddings_matrix)

        kg_embedding = K.dot(kg_word_embeddings, kg_relation_embeddings)

        # train word-to-kg embedding

        return K.dot(question_vector, kg_embedding)
