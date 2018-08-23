#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Aug 23, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Sample data to test the KBQA network
format based on LC-QuAD https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/train-data.json

'''
import pickle as pkl

QS = ["What is the river whose mouth is in deadsea?"]
AS = ["http://dbpedia.org/resource/Jordan_River"]

with open('aifb.pickle', 'rb') as f:
    data = pkl.load(f)

KB = data['A']
