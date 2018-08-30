#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 30, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Neural network model with R-GCN layer for KBQA

Based on Keras implementation of RGCN layer https://github.com/tkipf/relational-gcn

Question - text as the sequence of words (word index)
Answer - entity from KB (entity index)

'''
import sys

from rgcn_settings import *
from utils import *

class KBQA_RGCN:
    '''
    NN model for KBQA with R-GCN for KB embedding training
    '''
    def __init__(self):
        # load word embeddings with its vocabulary
        self.wordToIndex, self.indexToWord, self.wordToGlove = readGloveFile()

    def load_data(self, dataset):
        questions, answers = dataset
        num_samples = len(questions)
        assert num_samples == len(answers)

        # encode questions with word vocabulary and answers with entity vocabulary
        questions_data = []
        answers_data = []

        # iterate over samples
        for i in range(num_samples):
            # encode words of the question (ignore OOV words)
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]


def main(mode):
    '''
    Train model by running: python rgcn_kbqa2.py train
    '''
    model = KBQA_RGCN()
    # train on train split / test on test split
    dataset_split = mode

    # load data
    dataset = load_dataset(dataset_name, dataset_split)
    model.load_data(dataset)


if __name__ == '__main__':
    set_random_seed()
    main(sys.argv[1])
