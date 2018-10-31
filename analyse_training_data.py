#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 31, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Go through the dataset to detect the anomalies that may hinder learning
'''

from utils import load_dataset


def check_answer_distribution(dataset_name='lcquad', split='train'):
    '''
    Check distribution of answer entities in the training data
    '''
    # load data
    questions, answers = load_dataset(dataset_name, split)
    num_samples = len(questions)
    assert num_samples == len(answers)
    print('Loaded %s with %d %s samples' % (dataset_name, num_samples, split))

    answer_distribution = Counter([answer for question_answers in answers for answer in question_answers])
    print answer_distribution

# TODO resample training data to balance answer entities


if __name__ == '__main__':
    check_answer_distribution()
