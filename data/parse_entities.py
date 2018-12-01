#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 12, 2018

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Parse entities from lcquad entity span annotations file
'''
import json
import io

def parse_question_entities(path="./lcquad/lcquad.json"):
    '''
    Collect correct Entity Linking samples
    
    '''
    with io.open(path, "r", encoding="utf-8") as infile, io.open("els.txt", 'w', encoding="utf-8") as out:
        qs = json.load(infile)
        print ("%d total questions in lcquad" % (len(qs)))
        for q in qs:
            entities = q["entity mapping"]
            predicates = q["predicate mapping"]
            entities.extend(predicates)
            for entity in entities:
                # skip missing mentions per "label": "@@@" or "seq": "-1,-1"
                if entity['label'] and entity['label'] != "@@@":
                    if 'mappedBy' in entity.keys():
                        if entity['mappedBy'] != 'miss':
                            # strip URIs into labels
                            entity_label = entity['uri'].strip('/').split('/')[-1].strip('>').lower()
                            mention = entity['label'].lower()
                            # skip obvious exact string matches
                            if mention != entity_label:
                                # write out as: surface form mention \t URI label wo domain
                                out.write("%s\t%s\n"%(mention, entity_label))


def parse_dbpedia_entities(path="./predicates.txt"):
    with open(path, "r") as infile, open("predicates_labels.txt", 'w') as out:
        for line in infile:
            # line template http://creativecommons.org/ns#license;2
            entity_uri = ';'.join(line.split(';')[:-1])
            # out.write("%s\n"%(entity_uri))

            entity_label = entity_uri.strip('/').split('/')[-1].strip('>').lower()
            out.write("%s\n"%(entity_label))


if __name__ == '__main__':
    parse_dbpedia_entities()
