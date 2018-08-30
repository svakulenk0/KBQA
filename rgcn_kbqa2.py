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

from keras.preprocessing.text import text_to_word_sequence

from rgcn_settings import *
from utils import *


class KBQA_RGCN:
    '''
    NN model for KBQA with R-GCN for KB embedding training
    '''
    def __init__(self, load_word_embeddings=readGloveFile):
        # load word embeddings with its vocabulary into maps
        self.wordToIndex, self.indexToWord, self.wordToGlove = load_word_embeddings()
        # load entity vocabulary into a map
        self.entityToIndex = loadKB()

    def load_data(self, dataset):
        questions, answers = dataset
        num_samples = len(questions)
        assert num_samples == len(answers)

        # encode questions with word vocabulary and answers with entity vocabulary
        questions_data = []
        answers_data = []

        # iterate over samples
        for i in range(num_samples):
            # encode words in the question (ignore OOV words i.e. words without pre-trained embeddings)
            questions_sequence = [self.wordToIndex[word] for word in text_to_word_sequence(questions[i]) if word in self.wordToIndex]
            questions_data.append(questions_sequence)

            # encode all entities in the answer (make sure that all possible answer entities are indexed)
            answer_set = [self.entityToIndex[entity] for entity in answers[i]]
            answers_data.append(answer_set)

        # normalize length
        questions_data = np.asarray(pad_sequences(questions_data, padding='post'))
        answers_data = np.asarray(pad_sequences(answers_data, padding='post'))

        print("Loaded the dataset")
        # show dataset stats
        print("Maximum number of words in a question sequence: %d"%questions_data.shape[1])
        print("Maximum number of entities in an answer set: %d"%answers_data.shape[1])


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
