#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

lcquad corpus processing utils
'''
from random import shuffle

import json


def load_lcquad(fields, dataset_split='train', shuffled=False, limit=None):
    '''
    fields <list> items of the dataset, e.g. corrected_question etc.
    dataset_split <string> file pointer
    shuffle <boolean> reorder samples
    limit <int> number of samples
    '''
    # load the dataset
    with open("../data/lcquad_%s.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        dataset_size = len(qas)
        # print ("%d total QA pairs in lcquad %s" % (dataset_size, dataset_split))
        
        # shuffle indices within the range for a sample of size limit
        index_shuf = list(range(dataset_size))
        if shuffled:
            shuffle(index_shuf)

        # cut to the required sample size
        if limit:
            index_shuf = index_shuf[:limit]
        
        # collect all required fields for each sample
        samples = []
        for i in index_shuf:
            sample = []
            for field in fields:
                sample.append(qas[i][field])
            samples.append(sample)

    return samples
