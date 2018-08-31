#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 30, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>
'''
dataset_name = 'lcquad'

# architecture parameters
train_word_embeddings = True
# Question encoder GRU parameters
rnn_units = 500  # dimension of the GRUs output layer for the hidden question representation
# KB encoder R-GCN parameters
gc_units = 200
gc_bases = -1
l2norm = 0.

# training parameters
batch_size = 100
epochs = 10  # 10
learning_rate = 1e-3
