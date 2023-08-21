"""
CLOCQ_TEMPORAL service: carpo and port is 7778
"""

import requests
import globals
import time
import re
import json
import hashlib
#import KnowledgeGraphInterfacePublic as kg
import CLOCQInterfaceClient as kg

# Abreviations
# WDT = "http://www.wikidata.org/prop/direct/"
# P = "http://www.wikidata.org/prop/"
# WD = "http://www.wikidata.org/entity/"
# PS = "http://www.wikidata.org/prop/statement/"
# PQ = "http://www.wikidata.org/prop/qualifier/"
# WDS = "http://www.wikidata.org/entity/statement/"
# schema = "http://schema.org/"
# rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
# WIKIPEDIA_EN = "https://en.wikipedia.org/wiki/"
# PROI = "http://wikiba.se/ontology#WikibaseItem"
# PROQ = "http://wikiba.se/ontology#Quantity"
# PROT = "http://wikiba.se/ontology#Time"
clocq = kg.CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443")

def get_temporalfromclocq(fact_dic, pro):
    tempfact_dic = {}
    for sta in fact_dic:
        istemp = 0
        for item in fact_dic[sta]['ps']:
            if item[1]['id'] in pro:
                if globals.PROT in pro[item[1]['id']]["type"]:
                    istemp = 1
                    break
        for item in fact_dic[sta]['pq']:
            if item[0]['id'] in pro:
                if globals.PROT in pro[item[0]['id']]["type"]:
                    istemp = 1
                    break
        if istemp == 1:
            tempfact_dic[sta] = {}
            tempfact_dic[sta]['ps'] = fact_dic[sta]['ps'].copy()
            tempfact_dic[sta]['pq'] = fact_dic[sta]['pq'].copy()

    return tempfact_dic

#get temporal facts of target nodes
def get_wikidata_temptuplesfromclocq(ids, pro):
    #done = set()
    wiki_ids_facts = []
    for id in ids:
        wiki_ids_facts.append(clocq.get_neighborhood(id, include_labels=True, p=10000))
        #done.add(id)
    fact_dic = get_clocq_name_type_fact_dic(wiki_ids_facts)
    tempfact_dic = get_temporalfromclocq(fact_dic, pro)
    return tempfact_dic

#get temporal facts of target nodes
def get_wikidata_twohoptemptuplesfromclocq(ids, pro):
    done = set()
    wiki_ids_facts = []
    for id in ids:
        wiki_ids_facts.append(clocq.get_neighborhood_two_hop(id, include_labels=True, p=10000))
        done.add(id)
    fact_dic = get_clocq_name_type_fact_dic(wiki_ids_facts)
    print("\n\nlen(fact_dic)")
    print(len(fact_dic))
    tempfact_dic = get_temporalfromclocq(fact_dic, pro)
    return tempfact_dic

def get_wikidata_twohoptuplesfromclocq(wiki_ids):
    done = set()
    wiki_ids_facts = []
    for ids in wiki_ids:
        id1 = ids[0]
        if id1 in done:
            continue
        #score = ids[1]
        #text = ids[2]
        id1 = id1.strip()
        id2 = id1.lstrip('http://www.wikidata.org/entity/')
        #print("Expand Id")
        #print(id1, id2, score, text)
        wiki_ids_facts.append(clocq.get_neighborhood_two_hop(id2, include_labels=True, p=10000))
        done.add(id1)
    return get_clocq_name_type_fact_dic(wiki_ids_facts)

def get_wikidata_tuplesfromclocq(wiki_ids):
    done = set()
    wiki_ids_facts = []
    for ids in wiki_ids:
        id1 = ids[0]
        if id1 in done:
            continue
        #score = ids[1]
        #text = ids[2]
        id1 = id1.strip()
        id2 = id1.lstrip('http://www.wikidata.org/entity/')
        #print("Expand Id")
        #print(id1, id2, score, text)
        wiki_ids_facts.append(clocq.get_neighborhood(id2, include_labels=True, p=10000))
        done.add(id1)
    return get_clocq_name_type_fact_dic(wiki_ids_facts)

# def get_wikidata_twohoptuplesfromclocq(ids):
#     clocq = CLOCQ()
#     done = set()
#     wiki_ids_facts = []
#     for id in ids:
#         result = clocq.get_neighborhood(id, include_labels=True, p=10000)
#         wiki_ids_facts.append(result)
#         done.add(id)
#     return get_clocq_name_type_fact_dic(wiki_ids_facts)

def get_clocq_name_type_fact_dic(wiki_ids_facts):
    fact_dic = {}
    for id_facts in wiki_ids_facts:
        for item in id_facts:
            # get md5 hash of fact
            str2hash = ''
            for it in item:
                str2hash += it['id']
            #print (str2hash)
            md5hash = hashlib.md5(str2hash.encode()).hexdigest()
            statementid = item[0]['id'] + '-' + md5hash
            if statementid not in fact_dic:
                fact_dic[statementid] = {}
                fact_dic[statementid]['ps'] = []
                fact_dic[statementid]['pq'] = []
            fact_dic[statementid]['ps'].append((item[0], item[1], item[2]))
            pqn = int((len(item) - 3) / 2)
            if pqn > 0:
                for i in range(pqn):
                    pq_fact = (item[3 + i * 2], item[4 + i * 2])
                    if pq_fact not in fact_dic[statementid]['pq']:
                        fact_dic[statementid]['pq'].append(pq_fact)
    return fact_dic

def write_clocqtempspo_to_file(tempfact, tempspo_file, ids):
    fp = open(tempspo_file, 'w', encoding='utf-8')
    # for entities
    ENT_PATTERN = re.compile('^Q[0-9]+$')
    # for predicates
    PRE_PATTERN = re.compile('^P[0-9]+$')
    for sta in tempfact:
        t = tempfact[sta]['ps'][0]
        sub = t[0]['id'].strip('"')
        pre = t[1]['id'].strip('"')
        obj = t[2]['id'].strip('"')
        subname = sub
        objname = obj
        if isinstance(t[0]['label'], list):
            for item in t[0]['label']:
                if ENT_PATTERN.match(item) == None:
                    subname = item
                    break
        else:
            subname = t[0]['label'].strip('"')
        if isinstance(t[2]['label'], list):
            for item in t[2]['label']:
                if ENT_PATTERN.match(item) == None:
                    objname = item
                    break
        else:
            objname = t[2]['label'].strip('"')
        for id in ids:
            if sub == id:
                sub = "corner#" + sub
            if obj == id:
                obj = "corner#" + obj

        p = sub + "||" + subname + "||" + pre
        fp.write(sta + "-ps:" + "||" + p + "||" + pre + "||" + obj + "||" + str(objname) + '\n')
        for pqt in tempfact[sta]['pq']:
            pre = pqt[0]['id'].strip('"')
            obj = pqt[1]['id'].strip('"')
            objname = obj
            if isinstance(pqt[1]['label'], list):
                for item in pqt[1]['label']:
                    if ENT_PATTERN.match(item) == None:
                        objname = item
                        break
            else:
                objname = pqt[1]['label'].strip('"')
            for id in ids:
                if obj == id:
                    obj = "corner#" + obj
            fp.write(sta + "-pq:" + "||" + p + "||" + pre + "||" + obj + "||" + objname + '\n')
    fp.close()
    return

def write_clocqspo_to_file(fact, spo_file, wiki_ids):
    fp = open(spo_file, 'w', encoding='utf-8')
    # for entities
    ENT_PATTERN = re.compile('^Q[0-9]+$')
    # for predicates
    PRE_PATTERN = re.compile('^P[0-9]+$')
    for sta in fact:
        t = fact[sta]['ps'][0]
        sub = t[0]['id'].strip('"')
        pre = t[1]['id'].strip('"')
        obj = t[2]['id'].strip('"')
        subname = sub
        objname = obj

        # if ENT_PATTERN.match(sub) != None:
        #     subname = clocq.get_label(sub)
        # if ENT_PATTERN.match(obj) != None:
        #     objname = clocq.get_label(obj)

        if isinstance(t[0]['label'], list):
            for item in t[0]['label']:
                if ENT_PATTERN.match(item) == None:
                    subname = item
                    break
        else:
           subname = t[0]['label'].strip('"')
        if isinstance(t[2]['label'], list):
            for item in t[2]['label']:
                if ENT_PATTERN.match(item) == None:
                    objname = item
                    break
        else:
           objname = t[2]['label'].strip('"')
        for ids in wiki_ids:
            id1 = ids[0]
            score = ids[1]
            text = ids[2]
            id2 = id1.lstrip('http://www.wikidata.org/entity/')
            if sub == id2:
                sub = "corner#" + sub + "#" + str(score) + "#" + text
            if obj == id2:
                obj = "corner#" + obj + "#" + str(score) + "#" + text

        p = sub + "||" + subname + "||" + pre
        fp.write(sta + "-ps:" + "||" + p + "||" + pre + "||" + obj + "||" + str(objname) + '\n')
        for pqt in fact[sta]['pq']:
            pre = pqt[0]['id'].strip('"')
            obj = pqt[1]['id'].strip('"')
            objname = obj
            if isinstance(pqt[1]['label'], list):
                for item in pqt[1]['label']:
                    if ENT_PATTERN.match(item) == None:
                        objname = item
                        break
            else:
                objname = pqt[1]['label'].strip('"')
            for ids in wiki_ids:
                id1 = ids[0]
                score = ids[1]
                text = ids[2]
                id2 = id1.lstrip('http://www.wikidata.org/entity/')
                if obj == id2:
                    obj = "corner#" + obj + "#" + str(score) + "#" + text
            fp.write(sta + "-pq:" + "||" + p + "||" + pre + "||" + obj + "||" + objname + '\n')
    fp.close()
    return

