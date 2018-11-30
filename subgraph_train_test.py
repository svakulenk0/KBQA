#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 26, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Crawl GS subgraph for each question from DBpedia HDT
'''
import numpy as np
from subprocess import call, Popen, PIPE
import scipy.sparse as sp

from lcquad import load_lcquad
from index import IndexSearch


def generate_adj_sp(adjacencies, adj_shape, normalize=False, include_inverse=False):
    sp_adjacencies = []
    for edges in adjacencies:
        # split subject (row) and object (col) node URIs
        row, col = np.transpose(edges)

        # create adjacency matrix for this property
        data = np.ones(len(row), dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        if normalize:
            adj = normalize_adjacency_matrix(adj)
        sp_adjacencies.append(adj)

        # create adjacency matrix for inverse property
        if include_inverse:
            adj = sp.csr_matrix((data, (col, row)), shape=adj_shape, dtype=np.int8)
            if normalize:
                adj = normalize_adjacency_matrix(adj)
            sp_adjacencies.append(adj)

    return sp.hstack(sp_adjacencies, format="csr")


def generate_adj(subgraph):
    # collect edges separate for each property
    adjacencies = []
    current_p = None

    # parse subgraph triples
    entities = {}
    for triple in subgraph.split('\n'):
        # print (triple)
        s, p, o = triple.split()
        # switch to another property
        if p != current_p:
            if current_p:
                adjacencies.append(edges)
            current_p = p
            # create array to hold all edges per property
            edges = []
        # index
        if s not in entities.keys():
            entities[s] = len(entities)
        if o not in entities.keys():
            entities[o] = len(entities)
        edges.append(np.array([entities[s], entities[o]]))

    adjacencies.append(edges)

    adj_shape = (len(entities), len(entities))
    return generate_adj_sp(adjacencies, adj_shape), entities


def test_generate_adj():
    question_entities = ['232', '233']
    answer_entities = ['23', '43']
    subgraph = "232 34 23\n233 34 232\n234 34 43\n222 14 23\n232 14 23"
    A, entities = generate_adj(subgraph)
    print(A)
    print ([entities[entity_id] for entity_id in question_entities])
    print ([entities[entity_id] for entity_id in answer_entities])


if __name__ == '__main__':
    test_generate_adj()
