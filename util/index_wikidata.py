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
import io
import string

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from elasticsearch import TransportError

# setup
doc_type = 'terms'  # for mapping
KB = 'wikidata201809'
index_name = '%se' % KB
file_path = "../data/KB/terms_wikidata.txt"
ns_filter = "http://www.wikidata.org/entity"


# define streaming function
def uris_stream(index_name, file_path, doc_type, ns_filter=None):
    with io.open(file_path, "r", encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            # skip URIs if there is a filter set
            if ns_filter:
                if not line.startswith(ns_filter):
                    continue
            # line template http://www.wikidata.org/entity/P1660;101;has index case
            parse = line.split(';')
            
            entity_label = parse[2].strip()
            
            # skip malformed URIs
            if not entity_label or len(entity_label) > 150:
                continue

            # label preprocessing: remove punctuation, duplicate words, lowercase
            words = entity_label.split(' ')
            unique_words = []
            for word in words:
                # strip punctuation
                word = "".join([c for c in word if c not in string.punctuation])
                if word:
                    word = word.lower()
                    if word not in unique_words:
                        unique_words.append(word)
            entity_label = " ".join(unique_words)
            
            entity_uri = parse[0]
            count = parse[1]
            
            wd_id = entity_uri.strip('/').split('/')[-1]  # part of the WD URI
        
            data_dict = {'uri': entity_uri, 'label': entity_label,
                         'count': count, "id": i+1, 'label_exact': wd_id}
            
            print(data_dict)

            yield {"_index": index_name,
                   "_type": doc_type,
                   "_source": data_dict
                   }


es = Elasticsearch()
# iterate through input file in batches via streaming bulk
print("bulk indexing...")
try:
    for ok, response in streaming_bulk(es, actions=uris_stream(index_name, file_path, doc_type, ns_filter),
                                        chunk_size=200000):
        if not ok:
            # failure inserting
            print (response)
except TransportError as e:
    print(e.info)
    
print("Finished.")
