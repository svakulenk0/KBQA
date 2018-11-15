#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

lcquad corpus processing utils
'''

import json


def load_lcquad_qe(dataset_split='train'):
    QS = []
    ES = []
    empty_answers = 0
    with open("./data/lcquad_%s.json"%dataset_split, "r") as train_file:
        qas = json.load(train_file)
        
        print ("%d total QA pairs in lcquad %s" % (len(qas), dataset_split))

        for qa in qas:
            if qa['answers']:
                QS.append(qa['corrected_question'])
                ES.append(qa['entities'])
            else:
                empty_answers += 1
    
    print ("%d questions skipped because no answer was found" % empty_answers)
    return (QS, ES)
