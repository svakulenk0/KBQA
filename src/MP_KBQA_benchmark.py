#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Dec 9, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Message Passing for KBQA
'''

# path to KG
from subprocess import Popen, PIPE
hdt_lib_path = "/home/zola/Projects/hdt-cpp-molecules/libhdt"
hdt_file = './data/dbpedia2016-04en.hdt'
namespace = "http://dbpedia.org/"

# connect to indices
from index import IndexSearch
e_index = IndexSearch('dbpedia201604e')  # entity index
p_index = IndexSearch('dbpedia201604p')  # predicate index


# get a sample question from lcquad-train
from lcquad import load_lcquad

limit = 4000
samples = load_lcquad(fields=['corrected_question', 'entities', 'answers', 'sparql_template_id', 'sparql_query'],
                      dataset_split='train', shuffled=True, limit=limit)

# keep track of the covered templates
covered_templates = []

for sample in samples:
    question_o, correct_question_entities, answers, template_id, sparql_query = sample

    # skip questions with no answer found
    if not answers:
        continue

    # skip questions of the template we have already seen
    if template_id in covered_templates:
        continue

    # print('\n')
    print(template_id)
    covered_templates.append(template_id)
    # print (question_o)
    # print(sparql_query)
    # print(correct_question_entities)

    # parse the SPARQL query into the sequence of predicate expansions
    tripple_patterns = sparql_query[sparql_query.find("{")+1:sparql_query.find("}")].split('. ')
    # print('\n')
    # print(tripple_patterns)

    # collect entities and predicates separately for the intermediate nodes
    correct_intermediate_predicates = []
    correct_intermediate_entities = []
    correct_question_predicates = []
    correct_question_entities = []

    for pattern in tripple_patterns:
        if pattern:
            entities = []
            s, p, o = pattern.strip().split()
            if s[0] != '?':
                entities.append(s[1:-1])
            if o[0] != '?':
                entities.append(o[1:-1])
            p = p[1:-1]
            if '?uri' not in pattern:
                correct_intermediate_predicates.append(p)
                correct_intermediate_entities.extend(entities)
            else:
                correct_question_predicates.append(p)
                correct_question_entities.extend(entities)

    # get id of the answer entities
    answer_entities_ids = []
    for entity_uri in answers:
        matches = e_index.match_entities(entity_uri, match_by='uri')
        for match in matches:
            answer_entities_ids.append(str(matches[0]['_source']['id']))


    def get_KG_subgraph(seeds, predicates, nhops):
        # ./tools/hops data/dbpedia2016-04en.hdt -t "<http://dbpedia.org/resource/Delta_III>" -f "<http://dbpedia.org/ontology/manufacturer>" -n 1
        #  -f "<http://example.org/predicate1><http://example.org/predicate3>"
        p = Popen(["%s/tools/hops"%hdt_lib_path, "-t", "".join(["<%s>"%s for s in seeds]),
                   "-f", "".join(["<%s>"%p for p in predicates]),
                   '-p', namespace,
                   '-n', str(nhops), hdt_file], stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=hdt_lib_path)
        subgraph_str, err = p.communicate()
        # size of the subgraph (M triples)
        print("Extracted 1-hop subgraph size %d"%len(subgraph_str))
        return subgraph_str


    # reduce the subgraph to the top properties as a whitelist ("roads")
    # ! assume we know all correct predicates and entities
    # do the intermediate hop if required
    if correct_intermediate_predicates:
        top_entities, top_properties = correct_intermediate_entities, correct_intermediate_predicates
    else:
        top_entities, top_properties = correct_question_entities, correct_question_predicates
    # print(top_properties)

    subgraphs_str = get_KG_subgraph(top_entities, top_properties, nhops=1)


    # parse the subgraph into a sparse matrix
    import numpy as np
    import scipy.sparse as sp

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

    def parse_triples(triples_str, show_tripples=False):
        # extract the subgraph for the top properties
        adjacencies = {}
        # store mappings from local subgraph ids to global entity ids
        entities = {}
        re_entities = []
        # keep a list of selected edges for visualisation with networkx
        edge_list = []
        # iterate over triples
        # print('Parsing subgraphs..\n')
        for triple_str in triples_str.strip().split('\n'):
            terms = triple_str.split()
            s, p, o = terms
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
        return A, entities, re_entities, predicate_labels, edge_list

    A, entities, re_entities, predicate_labels, edge_list = parse_triples(subgraphs_str)


    # ## Message Passing


    # ! assume we know all correct entities

    # look up their ids in index
    question_entities_ids = []
    for entity_uri in top_entities:
        # if not in predicates check entities
        matches = e_index.match_entities(entity_uri, match_by='uri')
        if matches:
            question_entities_ids.append(str(matches[0]['_source']['id']))
        else:
            pass
            # print "%s not found" % entity_uri

    # print question_entities_ids
    top_entities_ids = question_entities_ids
    candidate_entities = entities.keys()

    # activations of entities
    # look up local entity id
    q_ids = [entities[entity_id] for entity_id in top_entities_ids if entity_id in candidate_entities]
    # assert len(q_ids) == len(top_entities)
    # graph activation vector TODO activate with the scores
    X = np.zeros(len(entities))
    X[q_ids] = 1

    # 1 hop
    # ! assume we know the correct predicate sequence activation
    # activate all predicates at once
    p_activations = np.ones(len(predicate_labels))
    # activate only the first entity
    # p_activations[[0, 1]] = 1
    # print p_activations.shape

    # activate adjacency matrices per predicate
    A1 = p_activations.T * A
    # print A1.shape

    # collect activations
    Y1 = np.zeros(len(entities))
    activations1 = []
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
    # print("%d entities activated"%n_activated)

    # draw top activated entities from the distribution
    if n_activated:    
        topn = 5
        top = Y1.argsort()[-n_activated:][::-1][:topn]
    #     print(top)
        # activation values
        # print Y1[top]
        
        # choose only the max activated entities
        # indices of the answers with maximum evidence support
        ind = np.argwhere(Y1 == np.amax(Y1)).T[0].tolist()
        # print("%d answers selected"%len(ind))
    #     print(ind)

        # all non-zero activations
    #     ind = np.argwhere(Y != 0)
        # print(ind)
        # indicate predicted answers
        Y1 = np.zeros(len(entities))
        Y1[ind] = 1
        Y = Y1


        activations1 = np.asarray(re_entities)[ind]
    #     print activations1
        
        # look up activated entities by ids
        activations1_labels = []
        for entity_id in activations1:
            matches = e_index.match_entities(int(entity_id), match_by='id', top=1)
            if matches:
              activations1_labels.append(matches[0]['_source']['uri'])
    #     print activations1_labels
        # print(activations1_labels[:topn])
        # activation values
        # print Y1[top]
    #     print("%d answers"%len(activations1))

        # did we hit the answer set already?
        hop1_answer = set(answer_entities_ids).issubset(set(activations1.tolist()))
        # print hop1_answer
    #     if hop1_answer:
        # print("%d correct answers"%len(answer_entities_ids))
        # print(answers[:topn])


    # 2 hop
    # check if we need the second hop to cover the remaining predicates and entities
    activations2 = []

    if correct_intermediate_predicates:
        # get next 1-hop subgraphs for all activated entities and the remaining predicates
        # choose properties for the second hop
        top_properties2 = correct_question_predicates
    #     print(top_properties2)
        activations1_labels.extend(correct_question_entities)
        subgraph_strs2 = get_KG_subgraph(activations1_labels, top_properties2, nhops=1)
    #     print(subgraphs_str2)
        # parse the subgraph into A
        A2, entities, re_entities2, predicate_labels2, edge_list2 = parse_triples(subgraph_strs2, show_tripples=False)


        # propagate activations
        candidate_entities = entities.keys()
    #     print(candidate_entities)
        X2 = np.zeros(len(entities))

        # activate entities selected at the previous hop
        top_entities2 = activations1
        # look up local entity id
        a_ids2 = [entities[entity_id] for entity_id in top_entities2 if entity_id in candidate_entities]
        # graph activation vector
        X2[a_ids2] = 1

        # and question entities activations for the 2nd hop
        if correct_question_entities:
            # look up their ids in the index
            question_entities_ids2 = []
            for entity_uri in correct_question_entities:
    #             print(entity_uri)
                # if not in predicates check entities
                matches = e_index.match_entities(entity_uri, match_by='uri')
                if matches:
                    question_entities_ids2.append(str(matches[0]['_source']['id']))
                else:
                    pass
                    # print "%s not found" % entity_uri
    #         print(question_entities_ids2)
            # look up local entity id
            q_ids2 = [entities[entity_id] for entity_id in question_entities_ids2 if entity_id in candidate_entities]
            # graph activation vector in the size of the previous activation layer to balance out
            X2[q_ids2] = len(a_ids2)
            # print(X)

        # ! assume we know the correct predicate sequence activation
        # activate all predicates at once
        p_activations2 = np.ones(len(predicate_labels2))
        # activate only the first entity
        # p_activations2[[0, 1]] = 1
        # print p_activations.shape

        A2 = p_activations2.T * A2
        # collect activations
        Y2 = np.zeros(len(entities))
        # activate adjacency matrices per predicate
        for i, a_p in enumerate(A2):
            # propagate from the previous activation layer
            y_p = X2*a_p
            # check if there is any signal through
            if sum(y_p) > 0:
                # add up activations
                Y2 += y_p
        
        # normalize activations
        Y2 -= len(a_ids2)
        
        # check output size
        assert Y2.shape[0] == len(entities)

        # check activated entities
        n_activated = np.nonzero(Y2)[0].shape[0]

        # draw top activated entities from the distribution
        if n_activated:
            Y = Y2
            n = n_activated
            top = Y2.argsort()[-n:][::-1]
            activations2 = np.asarray(re_entities2)[top]

            # look up activated entities by ids
            activations2_labels = []
            for entity_id in activations2:
                matches = e_index.match_entities(int(entity_id), match_by='id', top=1)
                if matches:
                  activations2_labels.append(matches[0]['_source']['uri'])
            
            # did we hit the answer set already?
            n_answers = len(answer_entities_ids)

    # indices of the answers with maximum evidence support
    # ind = np.argwhere(Y == np.amax(Y))
    # all non-zero activations
    ind = np.argwhere(Y > 0)
    # print(ind)

    # indicate predicted answers
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
    # report on error
    if n_errors > 0:
        print("%d errors"%n_errors)


print("All templates covered")