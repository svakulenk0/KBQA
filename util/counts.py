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

path = '../data/SUBTLEXus74286wordstextversion.txt'
test_q = "How many movies did Stanley Kubrick direct?"


def build_word_frequency_dict():
    wfd = {}
    i = 0
    # collect word frequencies into a dictionary
    with open(path, 'r') as file:
        for line in csv.reader(file, delimiter="\t"):
            if i != 0:
                # print line
                wfd[line[0]] = line[1]
                # break
            i += 1
    pickle.dump(wfd, open( "wfd.pkl", "wb" ) )


def test_wfd():
    wfd = pickle.load( open( "wfd.pkl", "rb" ) )
    for word in text_to_word_sequence(question):
        if word in wfd.keys():
            print(wfd[word])


if __name__ == '__main__':
    # build_word_frequency_dict()
    test_wfd()
