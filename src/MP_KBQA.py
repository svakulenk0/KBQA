#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 9, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Message Passing for KBQA
'''

from subprocess import Popen, PIPE

from index import IndexSearch
from lcquad import load_lcquad

# path to KG
hdt_lib_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt"
hdt_file = 'data/dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

# connect to indices
e_index = IndexSearch('dbpedia201604e')  # entity index
p_index = IndexSearch('dbpedia201604p')  # predicate index


# get a sample question from lcquad-train
limit = 1
samples = load_lcquad(fields=['corrected_question', 'entities', 'answers', 'sparql_template_id'],
                      dataset_split='train', shuffled=True, limit=limit)
question_o, correct_question_entities, answers, template_id = samples[0]

# we could not find answers to some of the lcquad questions
assert answers

print('\n')
print (question_o)

# drop duplicates
correct_question_entities = list(set(correct_question_entities))

# separate predicates from entities
correct_question_predicates = []
# get entity ids from index
question_entities_ids = []
question_predicates_ids = []
for entity_uri in correct_question_entities:
    # check predicates first
    matches = p_index.match_entities(entity_uri, match_by='uri')
    if matches:
        correct_question_predicates.append(entity_uri)
        question_predicates_ids.append(str(matches[0]['_source']['id']))
    else:
        # if not in predicates check entities
        matches = e_index.match_entities(entity_uri, match_by='uri')
        if matches:
            question_entities_ids.append(str(matches[0]['_source']['id']))
        else:
            print "%s not found" % entity_uri

correct_question_entities = list(set(correct_question_entities).difference(set(correct_question_predicates)))      
            
print (correct_question_entities)
print (correct_question_predicates)

# get id of the answer entities
answer_entities_ids = []
for entity_uri in answers:
    matches = e_index.match_entities(entity_uri, match_by='uri')
    for match in matches:
        answer_entities_ids.append(str(matches[0]['_source']['id']))

print('\n')
print(answers)


# ## Context Subgraph

# ! assume we know a correct seed entity
# TODO choose the most infrequent one to get a smaller subgraph, get frequency from the index
seed_entity = correct_question_entities[0]
print (seed_entity)
# skip it if we do not want to test predicate ranking
skip_this_step = True
if not skip_this_step:
    # request subgraph from the API (2 hops from the seed entity)
    print("Loading..")
    p = Popen(["%s/tools/hops"%hdt_lib_path, "-t", "<%s>"%seed_entity, '-p', namespace, '-n', '2', hdt_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=hdt_lib_path)
    subgraph_str, err = p.communicate()
    # size of the subgraph (M triples)
    print len(subgraph_str)


# ## Focus
# ! assume we know all correct predicates
top_properties = correct_question_predicates
print(top_properties)
# print(question_predicates_ids)
# reduce the subgraph to the top properties as a whitelist ("roads")
# print("Loading..")
# ./tools/hops data/dbpedia2016-04en.hdt -t "<http://dbpedia.org/resource/Delta_III>" -f "<http://dbpedia.org/ontology/manufacturer>" -n 1
#  -f "<http://example.org/predicate1><http://example.org/predicate3>"
p = Popen(["%s/tools/hops"%hdt_lib_path, "-t", "<%s>"%seed_entity,
           "-f", "".join(["<%s>"%p for p in top_properties]),
           '-n', '2', hdt_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=hdt_lib_path)
subgraph2_str, err = p.communicate()
# size of the subgraph (M triples)
# print subgraph2_str
print len(subgraph2_str)


# parse the subgraph into a sparse matrix
import scipy.sparse as sp
import numpy as np

def generate_adj_sp(adjacencies, adj_shape, normalize=False, include_inverse=False):
    # colect all predicate matrices separately into a list
    sp_adjacencies = []
    for p, edges in adjacencies.items():
        # split subject (row) and object (col) node URIs
        n_edges = len(edges)
        row, col = np.transpose(edges)
        
        # duplicate edges in the opposite direction
        if include_inverse:
            _row = np.hstack([row, col])
            col = np.hstack([col, row])
            row = _row
            n_edges *= 2
        
        # create adjacency matrix for this predicate TODO initialise matrix with predicate scores
        data = np.ones(n_edges, dtype=np.int8)
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)
        
        if normalize:
            adj = normalize_adjacency_matrix(adj)

        sp_adjacencies.append(adj)
    
    return np.asarray(sp_adjacencies)


# extract the subgraph for the top properties
show_tripples = False
adjacencies = {}

# store mappings from local subgraph ids to global entity ids
entities = {}
re_entities = []
# keep a list of selected edges for visualisation with networkx
edge_list = []
# iterate over triples
print('Parsing subgraph..\n')
for triple_str in subgraph2_str.strip().split('\n'):
    terms = triple_str.split()
    s, p, o = terms
#     print p
    # select only triples with one of the top predicates
    if p in question_predicates_ids:
            # print out selected subgraph triples
        # if show_tripples:
        #     highlight_triple = []
        #     highlighted = False
        #     for term in terms:
        #         if term in question_entities_ids:
        #             highlight_triple.append("\x1b[31m%s\x1b[0m"%term)
        #             highlighted = True
        #         elif term in answer_entities_ids:
        #             highlight_triple.append("\x1b[32m%s\x1b[0m"%term)
        #             highlighted = True
        #         else:
        #             highlight_triple.append(term)
        #     if highlighted:
        #         print ' '.join(highlight_triple)

        # index
        if s not in entities.keys():
            entities[s] = len(entities)
            re_entities.append(s)
        if o not in entities.keys():
            entities[o] = len(entities)
            re_entities.append(o)

        edge_list.append(' '.join([s, o]))
        # collect all edges per predicate
        edge = np.array([entities[s], entities[o]])
        if p not in adjacencies.keys():
            adjacencies[p] = []
        adjacencies[p].append(edge)
        
adj_shape = (len(entities), len(entities))
# assuming the graph is undirected wo self-loops
# generate a list of adjacency matrices per predicate
# print adjacencies
A = generate_adj_sp(adjacencies, adj_shape, include_inverse=True)
# print(A.shape)
# look up predicate sequence labels in the predicate index
predicate_labels = [p_index.match_entities(p_id, match_by='id', top=1)[0]['_source']['label'] for p_id in adjacencies.keys()]
# check adjacency size
assert len(A) == len(top_properties)

candidate_entities = entities.keys()
# print(candidate_entities)
# print(answer_entities_ids)
# make sure that we selected a correct subgraph TODO backpropagate
assert set(answer_entities_ids).issubset(set(candidate_entities))
# assert set(question_entities_ids).issubset(set(candidate_entities))

# show size of the subgraph
print("\nSubgraph:")
print("%d entities"%len(entities))
# print("%d edges"%len(G.edges()))
print("%d predicates"%len(top_properties))
print (predicate_labels)


# ## Message Passing

# ! assume we know all correct entities
top_entities = question_entities_ids
# activations of entities
# look up local entity id
q_ids = [entities[entity_id] for entity_id in top_entities if entity_id in candidate_entities]
assert len(q_ids) == len(top_entities)
# graph activation vector TODO activate with the scores
X = np.zeros(len(entities))
X[q_ids] = 1
# print(X)
print("%d entities activated"%len(q_ids))


# 1 hop
# ! assume we know the correct predicate sequence activation
p_activations = np.array([1, 1])
# print p_activations.shape
# activate adjacency matrices per predicate
A1 = p_activations.T * A
# print A1.shape

# collect activations
Y1 = np.zeros(len(entities))
activations1, activations2 = [], []
for i, a_p in enumerate(A1):
    # activate current adjacency matrix via input propagation
    y_p = X*a_p
    # check if there is any signal through
    if sum(y_p) > 0:
        # add up activations
        Y1 += y_p

# check output size
assert Y1.shape[0] == len(entities)

# check activated entities
n_activated = np.nonzero(Y1)[0].shape[0]
print("%d entities activated"%n_activated)

# draw top activated entities from the distribution
if n_activated:
    Y = Y1
    n = n_activated
    top = Y1.argsort()[-n:][::-1]
    activations1 = np.asarray(re_entities)[top]
#     print activations1
    
    # look up activated entities by ids
    activations1_labels = []
    for entity_id in activations1:
        matches = e_index.match_entities(int(entity_id), match_by='id', top=1)
        if matches:
          activations1_labels.append(matches[0]['_source']['uri'])
#     print activations1_labels
    topn = 5
    print(activations1_labels[:topn])
    # activation values
    print Y1[top]
    print("%d answers"%len(activations1))

    # did we hit the answer set already?
    hop1_answer = set(answer_entities_ids).issubset(set(activations1))
    print hop1_answer
    if hop1_answer:
        print("%d correct answers"%len(answer_entities_ids))
        print(answers)


# 2 hop
# activate the rest of the predicates
p_activations2 = 1 - p_activations
# print(p_activations2)
# continue propagation if there are still unused predicates
if sum(p_activations2) > 0:
    A2 = p_activations2.T * A
    # collect activations
    Y2 = np.zeros(len(entities))
    # activate adjacency matrices per predicate
    for i, a_p in enumerate(A2):
        # propagate from the previous activation layer
        y_p = Y1*a_p
        # check if there is any signal through
        if sum(y_p) > 0:
            # add up activations
            Y2 += y_p
        
    # check output size
    assert Y2.shape[0] == len(entities)

    # check activated entities
    n_activated = np.nonzero(Y2)[0].shape[0]
    print("%d entities activated"%n_activated)

    # draw top activated entities from the distribution
    if n_activated:
        Y = Y2
        n = n_activated
        top = Y2.argsort()[-n:][::-1]
        activations2 = np.asarray(re_entities)[top]
        
        print(activations2)

        # look up activated entities by ids
        activations2_labels = [e_index.match_entities(entity_id, match_by='id', top=1)[0]['_source']['label'] for entity_id in activations2]
        topn = 5
        print(activations2_labels[:topn])
        # activation values
        print Y2[top[:topn]]

        # did we hit the answer set already?
        print(set(answer_entities_ids).issubset(set(activations2)))
        n_answers = len(answer_entities_ids)
        print("%d correct answers"%n_answers)
        assert n_activated == n_answers
        print(answers)


# draw the propagation answer graph
import networkx as nx
# parse the subgraph into networkx graph
G = nx.parse_edgelist(edge_list, delimiter=' ', nodetype=str, create_using=nx.DiGraph())
# 1 hop activations
# answer_graph = G.subgraph(list(activations1)+top_entities)
# all activations 1 hop and 2 hop
answer_graph = G.subgraph(list(activations1)+list(activations2)+top_entities)

# error estimation
# print Y
# choose answers with maximum evidence support
# indices of these answers
ind = np.argwhere(Y == np.amax(Y))
# print(ind)
# indicate the highly predicted entities as answers
Y_pr = np.zeros(len(entities))
Y_pr[ind] = 1
# print(Y_pr)

# produce correct answers vector
Y_gs = np.zeros(len(entities))
# translate correct answers ids to local subgraph ids
a_ids = [entities[entity_id] for entity_id in answer_entities_ids if entity_id in candidate_entities]
Y_gs[a_ids] = 1
# print(Y_gs)

# check errors as difference between distributions
error_vector = Y_gs - Y_pr
# print (error_vector)
n_errors = len(np.nonzero(error_vector)[0])
print("%d errors"%n_errors)                     
# TODO backpropagate
