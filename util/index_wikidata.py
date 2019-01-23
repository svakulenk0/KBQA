#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jan 18, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Index WikiData KG entities in ES

To restart indexing:

1. Delete previous index
curl -X DELETE "localhost:9200/wikidata201809e"

2. Put mapping (see mapping.json file)
curl -X PUT "localhost:9200/wikidata201809e" -H 'Content-Type: application/json' -d'
...

3. Run this script. Check progress via
curl -XGET "localhost:9200/wikidata201809e/_count"

'''

# index entities

KB = 'wikidata201809'
file_path = "../data/KB/terms_wikidata.txt"
ns_filter = "http://www.wikidata.org"
ENDPOINT = 'https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&languages=en&format=json&ids=' # WD API

import io
import re, string
import requests

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError

# pre-processing
def parse_label(entity_label):
    entity_label = " ".join(re.sub('([a-z])([A-Z])', r'\1 \2', entity_label.rstrip().lstrip()).split())
    words = entity_label.split(' ')
    unique_words = []
    for word in words:
        # strip punctuation
        word = "".join([" " if c in string.punctuation else c for c in word])
        if word:
            word = word.lower()
            if word not in unique_words:
                unique_words.append(word)
    entity_label = " ".join(unique_words)
    return entity_label


# setup
es = Elasticsearch()
doc_type = 'terms'  # for mapping

# define streaming function
def uris_stream(index_name, file_path, doc_type, ns_filter=None):
    with io.open(file_path, "r", encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if ns_filter and not line.startswith(ns_filter):
                continue
            parse = line.split(';')
            entity_uri = parse[0]
            count = parse[1]
            wd_id = entity_uri.strip('/').split('/')[-1]  # part of the WD URI
            entity_label = parse[2].strip()
            if not entity_label:
                response = requests.get(ENDPOINT+wd_id).json()['entities'][wd_id]
                if 'labels' in response:
                    labels = response['labels']
                    if labels:
                        entity_label = labels['en']['value']
            entity_label = parse_label(entity_label)
            
            data_dict = {'uri': entity_uri, 'label': entity_label,
                         'count': count, "id": i+1, 'label_exact': wd_id}

            yield {"_index": index_name,
                   "_type": doc_type,
                   "_source": data_dict
                   }

# index entities
index_name = '%se' % KB  # entities index

# iterate through input file in batches via streaming bulk
print("bulk indexing...")
try:
    for ok, response in streaming_bulk(es, actions=uris_stream(index_name, file_path, doc_type, ns_filter),
                                       chunk_size=100000):
        if not ok:
            # failure inserting
            print (response)
except TransportError as e:
    print(e.info)
    
print("Finished.")