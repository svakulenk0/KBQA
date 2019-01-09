#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Jan 8, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Evaluate entity linking performance and store annotations
'''

# setup
dataset_name = 'lcquad'

import os
os.chdir('/home/zola/Projects/temp/KBQA/util')
from setup import IndexSearch, Mongo_Connector, load_embeddings

e_vectors = load_embeddings('fasttext_e_labels')
e_index = IndexSearch('dbpedia201604e')
mongo = Mongo_Connector('kbqa', dataset_name)

# match and save matched entity URIs to MongoDB
loaded = False

limit = None
string_cutoff = 50  # maximum number of candidate entities per mention
semantic_cutoff = 1000
max_degree = 50000

# path to KG relations
from hdt import HDTDocument
hdt_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt/data/"
hdt_file = 'dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

import numpy as np
print("Entity linking...")
def entity_linking(spans_field, save, show_errors=True, add_nieghbours=True, lookup_embeddings=False):
    # iterate over the cursor
    cursor = mongo.get_sample(limit=limit)
    count = 0
    # hold macro-average stats for the model performance over the samples
    ps, rs, fs = [], [], []
    with cursor:
        for doc in cursor:
            if 'entity_ids_guess' not in doc:
                correct_uris = doc['entity_uris']
                print(set(correct_uris))
                # get entity spans
                e_spans = doc[spans_field]
        #         e_spans = doc[spans_field+'_guess']
            #     print(e_spans)
                # get entity matches TODO save scores
                top_ids = []
                top_entities = {}
                for span in e_spans:
                    print("Span: %s"%span)
                    print("Index lookup..")
                    guessed_labels, guessed_ids, look_up_ids = [], [], []
                    for match in e_index.match_label(span, top=string_cutoff):
                        label = match['_source']['label_exact']
                        degree = match['_source']['count']
        #                 print(degree)
                        _id = match['_source']['id']
                        # avoid expanding heavy hitters
                        if int(degree) < max_degree:
                            look_up_ids.append(_id)
                        guessed_ids.append(_id)
                        if label not in guessed_labels:
                            guessed_labels.append(label)
                        uri = match['_source']['uri']
        #                 print(uri)

                    print("%d candidate labels"%len(guessed_labels))
                    if add_nieghbours:
                        print("KG lookup..")
                        kg = HDTDocument(hdt_path+hdt_file)
                        kg.configure_hops(1, [], namespace, True)
                        entities, predicate_ids, adjacencies = kg.compute_hops(look_up_ids)
                        kg.remove()
                        # look up labels
                        for e_id in entities:
                            match = e_index.look_up_by_id(e_id)
                            if match:
                                label = match[0]['_source']['label_exact']
                                if label not in guessed_labels:
                                    guessed_labels.append(label)
                        guessed_ids.extend(entities)

                    # score with embeddings
                    guessed_labels = [label for label in guessed_labels if label in e_vectors]
                    print("%d candidate labels"%len(guessed_labels))
                    if guessed_labels and lookup_embeddings:
                        print("Embeddings lookup..")
                        dists = e_vectors.distance(span, [label for label in guessed_labels if label in e_vectors])
                        top = np.argsort(dists)[:semantic_cutoff].tolist()
                        top_labels = [guessed_labels[i] for i in top]
                        print("selected labels: %s"%top_labels)
                        print("Index lookup..")
                        top_entities[span] = []
                        for i, label in enumerate(top_labels):
                            print(label)
                            for match in e_index.look_up_by_label(label):
                                distance = float(dists[top[i]])
                                degree = match['_source']['count']
                                _id = match['_source']['id']
                                uri = match['_source']['uri']
                                print(uri)
                                top_entities[span].append({'rank': i+1, 'distance': distance, 'degree': degree, 'id': _id, 'uri': uri})
                                top_ids.append(_id)
                    else:
                        top_labels = guessed_labels
                        top_ids.extend(guessed_ids)

                # evaluate against the correct entity ids
                top_ids = list(set(top_ids))
                correct_ids = set(doc['entity_ids'])
                n_hits = len(correct_ids & set(top_ids))
                try:
                    r = float(n_hits) / len(correct_ids)
                except ZeroDivisionError:\
                    print(doc['question'])
                try:
                    p = float(n_hits) / len(top_ids)
                except ZeroDivisionError:
                    p = 0
                try:
                    f = 2 * p * r / (p + r)
                except ZeroDivisionError:
                    f = 0
                print("P: %.2f R: %.2f F: %.2f"%(p, r, f))

                # add stats
                ps.append(p)
                rs.append(r)
                fs.append(f)

                # save to MongoDB
                if save:
                    doc['entity_ids_guess'] = top_ids
                    mongo.col.update_one({'_id': doc['_id']}, {"$set": doc}, upsert=True)
                    count += 1

    print("P: %.2f R: %.2f F: %.2f"%(np.mean(ps), np.mean(rs), np.mean(fs)))
    print("Fin. Results for %d questions"%len(ps))    
    if save:
        print("%d documents annotated with entity ids guess"%count)

    
if not loaded:
    # evaluate entity linking on extracted entity spans
    entity_linking(spans_field='entity_spans', save=True)
