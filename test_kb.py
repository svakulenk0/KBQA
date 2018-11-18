#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Parse KG entities

Following https://qbox.io/blog/building-an-elasticsearch-index-with-python
'''
import csv

path = "./data/entitiesWithObjectsURIs.txt"

with open(path, "r") as in_file:

    bulk_data = []

    for i, line in enumerate(in_file):
        # line template http://creativecommons.org/ns#license;2
        parse = line.split(';')
        entity_uri = ';'.join(parse[:-1])
        count = parse[-1].strip()
        entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')

        data_dict = {'uri': entity_uri, 'label': entity_label, 'count': count, 'id': i+1}

        # print data_dict
        #     data_dict[header[i]] = row[i]
    # op_dict = {
    #     "index": {
    #         "_index": INDEX_NAME, 
    #         "_type": TYPE_NAME, 
    #         "_id": data_dict[ID_FIELD]
    #     }
    # }
    # bulk_data.append(op_dict)
    bulk_data.append(data_dict)
    print len(bulk_data)
     
    # header = csv_file_object.next()
    # header = [item.lower() for item in header]
    # print header
    # print len(infile.read())
    # for i, l in enumerate(f):
    #     pass
    # print i + 1

    # for line in infile:
    #     # line template http://creativecommons.org/ns#license;2
    #     parse = line.split(';')
    #     entity_uri = ';'.join(parse[:-1])
    #     count = parse[-1].strip()
    #     entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower().strip('ns#')
