#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 6, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Analyse KG
'''
from collections import Counter


def sort_words_by_frequency(file_path='./data/DBpedia_KGlove_labels.txt'):
    '''
    Split all entity labels into words, sort by count and write into a file starting from the least common
    '''
    # maintain counter
    words = Counter()
    
    # read all labels
    with open(file_path) as file:
        for label in file:
            # split label into words
            words = label.split('_')
            print words


if __name__ == '__main__':
    sort_words_by_frequency()
