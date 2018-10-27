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

    def __init__(self, kg_word_embeddings_matrix, kg_relation_embeddings_matrix, word_embs_dim, kg_embeddings_dim, **kwargs):
        self.kg_word_embeddings_matrix = kg_word_embeddings_matrix
        self.kg_relation_embeddings_matrix = kg_relation_embeddings_matrix
        self.word_embs_dim = word_embs_dim
        self.kg_embeddings_dim = kg_embeddings_dim

        super(EntityLinking, self).__init__(**kwargs)

    def build(self, input_shape):
        kg_word_embeddings = K.variable(self.kg_word_embeddings_matrix.T)
        kg_relation_embeddings = K.variable(self.kg_relation_embeddings_matrix)
        self.kg_embedding = K.dot(kg_word_embeddings, kg_relation_embeddings)
        
        # Create a trainable weight variable for word-to-kg embedding
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.kg_embeddings_dim, self.kg_embeddings_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(EntityLinking, self).build(input_shape)  # Be sure to call this at the end

    def call(self, question_words_embeddings, mask=None):
        # multiply with weights kernel
        kg_embedding = K.dot(self.kg_embedding, self.kernel)
        question_kg_embedding = K.dot(question_words_embeddings, kg_embedding)
        # print K.int_shape(question_kg_embedding)
        return question_kg_embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.kg_embeddings_dim)

    def get_config(self):
        base_config = super(EntityLinking, self).get_config()
        # base_config['output_dim'] = self.kg_embeddings_dim
        # return base_config
        config = {'kg_word_embeddings_matrix': self.kg_word_embeddings_matrix, 
                  'kg_relation_embeddings_matrix': self.kg_relation_embeddings_matrix,
                  'word_embs_dim': self.word_embs_dim,
                  'kg_embeddings_dim': self.kg_embeddings_dim}
        return dict(list(base_config.items()) + list(config.items()))
