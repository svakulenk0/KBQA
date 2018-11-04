#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 27, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Entity Linking layer trained to align word and entity embeddings.

'''
import numpy as np

from keras import backend as K
from keras.engine.topology import Layer


class EntityLinking(Layer):

    def __init__(self, kg_word_embeddings_matrix, kg_relation_embeddings_matrix, kg_word_embeddings_initializer,
                 word_embs_dim, kg_embeddings_dim, num_entities, **kwargs):
        # when loaded from config matrices are again lists not numpy arrays
        self.kg_word_embeddings_matrix = np.asarray(kg_word_embeddings_matrix, dtype=K.floatx())
        self.kg_relation_embeddings_matrix = np.asarray(kg_relation_embeddings_matrix, dtype=K.floatx())
        self.kg_word_embeddings_initializer = kg_word_embeddings_initializer
        self.word_embs_dim = word_embs_dim
        self.kg_embeddings_dim = kg_embeddings_dim
        self.num_entities = num_entities

        super(EntityLinking, self).__init__(**kwargs)

    def build(self, input_shape):
        # ValueError: Cannot create a tensor proto whose content is larger than 2GB.
        # kg_word_embeddings = K.variable(self.kg_word_embeddings_matrix.T)

        # work-around
        kg_word_embeddings = K.variable(self.kg_word_embeddings_initializer)

        kg_relation_embeddings = K.variable(self.kg_relation_embeddings_matrix)
        self.kg_embedding = K.dot(kg_word_embeddings, kg_relation_embeddings)
        
        # Create a trainable weight variable for word-to-kg embedding
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.kg_embeddings_dim, self.kg_embeddings_dim),
                                      # shape=(self.word_embs_dim, self.word_embs_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(EntityLinking, self).build(input_shape)  # Be sure to call this at the end

    def call(self, question_words_embeddings, mask=None):
        # multiply with weights kernel
        # question_words_embeddings = K.dot(question_words_embeddings, self.kernel)  # model 2 trainable
        
        kg_embedding = K.dot(self.kg_embedding, self.kernel)
        question_kg_embedding = K.dot(question_words_embeddings, kg_embedding)
        # print K.int_shape(question_kg_embedding)
        return question_kg_embedding  # model 3

        # return K.dot(question_words_embeddings, self.kernel)  # model 1 (baseline) trainable
        # return K.dot(question_words_embeddings, K.variable(self.kg_word_embeddings_matrix.T))  # model 2

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], input_shape[1], self.word_embs_dim)  # model 1
        # return (input_shape[0], input_shape[1], self.num_entities)  # model 2
        return (input_shape[0], input_shape[1], self.kg_embeddings_dim)  # model 3

    def get_config(self):
        base_config = super(EntityLinking, self).get_config()
        # base_config['output_dim'] = self.kg_embeddings_dim
        # return base_config
        config = {'kg_word_embeddings_matrix': self.kg_word_embeddings_matrix, 
                  'kg_relation_embeddings_matrix': self.kg_relation_embeddings_matrix,
                  'num_entities': self.num_entities,
                  'word_embs_dim': self.word_embs_dim,
                  'kg_embeddings_dim': self.kg_embeddings_dim}
        return dict(list(base_config.items()) + list(config.items()))
