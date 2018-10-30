#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>
'''
dataset_name = 'lcquad'

# architecture parameters
rnn_units = 100  # dimension of the GRUs output layer for the hidden question representation
output_vector = 'one-hot'  # output as a 'one-hot' or KG 'embedding' vector

# dataset parameters
max_question_words = 24  # maximum number of words in a question

# training parameters
batch_size = 32
epochs = 50  # 10
learning_rate = 1e-3
