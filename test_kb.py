#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Parse KG entities
'''
path = "./data/entitiesWithObjectsURIs.txt"

with open(path, "r") as f:
    # print len(infile.read())
    for i, l in enumerate(f):
        pass
    print i + 1

    # for line in infile:
    #     # line template http://creativecommons.org/ns#license;2
    #     parse = line.split(';')
    #     entity_uri = ';'.join(parse[:-1])
    #     count = parse[-1].strip()
    #     entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')
