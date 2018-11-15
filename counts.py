#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 15, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Exploit frequency counts
'''

import csv
import pickle

from keras.preprocessing.text import text_to_word_sequence

from lcquad import load_lcquad_qe

path = './data/SUBTLEXus74286wordstextversion.txt'
test_q = "How many movies did Stanley Kubrick direct?"


def build_word_frequency_dict():
    wfd = {}
    i = 0
    # collect word frequencies into a dictionary
    with open(path, 'r') as file:
        for line in csv.reader(file, delimiter="\t"):
            if i != 0:
                # print line
                wfd[line[0].lower()] = line[1]
                # break
            i += 1
    pickle.dump(wfd, open( "wfd.pkl", "wb" ) )


def test_wfd(question=test_q):
    wfd = pickle.load( open( "wfd.pkl", "rb" ) )
    for word in text_to_word_sequence(question):
        if word in wfd.keys():
            print(wfd[word])
        else:
            print word


def test_wfd_on_lcquad(question=test_q, limit=10):
    wfd = pickle.load( open( "wfd.pkl", "rb" ) )

    questions, correct_question_entities = load_lcquad_qe()

    # iterate over questions
    for i, question in enumerate(questions):
        print question
        for word in text_to_word_sequence(question, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\''):
            if word in wfd.keys():
                print(wfd[word])
            else:
                print word
        if i == limit:
            break


if __name__ == '__main__':
    # build_word_frequency_dict()
    test_wfd_on_lcquad()
