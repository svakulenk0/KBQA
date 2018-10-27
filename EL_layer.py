#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 27, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Entity Linking layer trained to align word and entity embeddings.

'''
from keras import backend as K
from keras.engine.topology import Layer


class EntityLinking(Layer):

    def __init__(self, kg_word_embeddings_matrix, kg_relation_embeddings_matrix, output_dim, **kwargs):
        self.kg_word_embeddings_matrix = kg_word_embeddings_matrix
        self.kg_relation_embeddings_matrix = kg_relation_embeddings_matrix
        self.output_dim = output_dim

        super(EntityLinking, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        pass

    def call(self, question_words_embeddings, mask=None):
        kg_word_embeddings = K.variable(self.kg_word_embeddings_matrix.T)
        kg_relation_embeddings = K.variable(self.kg_relation_embeddings_matrix)
        kg_embedding = K.dot(kg_word_embeddings, kg_relation_embeddings)
        # train word-to-kg embedding
        # TODO weights
        # return K.dot(question_words_embeddings, K.variable(self.kg_word_embeddings_matrix.T))
        return K.dot(question_words_embeddings, kg_embedding)

    def get_output_shape_for(self, input_shape):
        # return (input_shape[0], input_shape[1], 27293)
        return (input_shape[0], input_shape[1], self.output_dim)

    # def get_config(self):
    #     pass
