#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>
'''
dataset_name = 'lcquad'

# architecture parameters
# n_words = 24  # maximum number of words in a question
rnn_units = 500  # dimension of the GRUs output layer for the hidden question representation

# training parameters
batch_size = 32
epochs = 50  # 10
learning_rate = 1e-3
