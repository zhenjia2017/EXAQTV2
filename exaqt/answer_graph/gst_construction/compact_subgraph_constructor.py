"""Script to get top-g GSTs from top-f scored facts for questions.
"""
import copy
import networkx as nx
import time
import os
import signal
import nltk
from pathlib import Path
nltk.download('stopwords')

from nltk.stem import PorterStemmer
PS = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
LC = LancasterStemmer()
from nltk.stem import SnowballStemmer
SB = SnowballStemmer("english")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from exaqt.answer_graph.gst_construction.get_GST import call_main_rawGST
from exaqt.library.utils import get_logger

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient

#prepare data...
print ("\n\nPrepare data and start...")
MAX_TIME = 300

def replace_symbols_in_relation(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace(':', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'s', ' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('/', ' ')
    s = s.replace('\\', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('.', ' ')
    s = s.replace(',', ' ')
    s = s.replace('\"', ' ')
    s = s.strip()
    return s

def replace_symbols_in_question(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'',' ')
    s = s.replace('\n',' ')
    s = s.replace('?', ' ')
    s = s.strip(',')
    s = s.rstrip('.')
    s = s.strip()
    return s

def property_type_label_alias(pre, pro_info):
    type = ""
    label = ""
    alias = []
    if pre in pro_info.keys():
        type = pro_info[pre]['type']
        label = pro_info[pre]['label']
        altLabel = pro_info[pre]['altLabel']
        if altLabel.find(", ") >= 0:
            alias = altLabel.split(", ")
        elif len(altLabel) > 0:
            alias = [altLabel]
    return type, label, alias

def build_graph_from_triple_edges(unique_SPO_dict, proinfo, sta_score):
    G = nx.DiGraph()
    pred_count = {}
    for (n1, n2, n3) in unique_SPO_dict:
        n2_id = n2.split("#")[0]
        n2_sta = n2.split("#")[1]
        sta_weight = sta_score[n2_sta]
        n22t, n22l, n22a = property_type_label_alias(n2_id, proinfo)
        n1_name = unique_SPO_dict[(n1, n2, n3)]['name_n1']
        n3_name = unique_SPO_dict[(n1, n2, n3)]['name_n3'].replace('T00:00:00Z','')
        n11 = n1_name + "::Entity::" + n1
        if n22l not in pred_count:
            pred_count[n22l] = 1
        else:
            pred_count[n22l] = pred_count[n22l] + 1
        n22 = n22l + "::Predicate::" + n2_sta + "::" + str(pred_count[n22l])
        n33 = n3_name + "::Entity::" + n3
        if n11 not in G.nodes():
            n1_alias = []
            G.add_node(n11, id=n1, alias=n1_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_n1'],
                       matched=unique_SPO_dict[(n1, n2, n3)]['matched_n1'])

        if n22 not in G.nodes():
            G.add_node(n22, id=n2_id, alias=n22a, type=n22t, weight=0.0, matched='')

        if n33 not in G.nodes():
            n3_alias = []
            G.add_node(n33, id=n3, alias=n3_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_n3'],
                       matched=unique_SPO_dict[(n1, n2, n3)]['matched_n3'])

        G.add_edge(n11, n22, weight=sta_weight, wlist=[sta_weight], etype='Triple')
        G.add_edge(n22, n33, weight=sta_weight, wlist=[sta_weight], etype='Triple')

        # ADD qualifier nodes edges

        if 'qualifier' in unique_SPO_dict[(n1, n2, n3)]:
            for qualct in range (0, len (unique_SPO_dict[(n1, n2, n3)]['qualifier'])):
                qual = unique_SPO_dict[(n1, n2, n3)]['qualifier'][qualct]
                #qn2 = qual[0]
                qn2_id = qual[0].split("#")[0]
                qn3_id = qual[1]
                qn3_name = qual[2].replace('T00:00:00Z','')
                #qn2,qn3_id,qn3_name
                qn22t, qn22l, qn22a = property_type_label_alias(qn2_id, proinfo)
                if qn22l not in pred_count:
                    pred_count[qn22l] = 1
                else:
                    pred_count[qn22l] = pred_count[qn22l] + 1
                qn22 = qn22l + "::Predicate::" + qual[0].split("#")[1] + "::" + str(pred_count[qn22l])

                qn33 = qn3_name + "::Entity::" + qn3_id

                if qn22 not in G.nodes ():
                    G.add_node (qn22, id=qn2_id, alias=qn22a, type=qn22t, weight=0.0, matched='')

                if qn33 not in G.nodes ():
                    qn33_alias = []
                    G.add_node(qn33, id=qn3_id ,alias=qn33_alias, weight=unique_SPO_dict[(n1, n2, n3)]['score_qn3'][qualct],
                                matched=unique_SPO_dict[(n1, n2, n3)]['matched_qn3'][qualct])

                G.add_edge (n22, qn22, weight=sta_weight, wlist=[sta_weight], etype='Triple')
                G.add_edge (qn22, qn33, weight=sta_weight, wlist=[sta_weight], etype='Triple')

    return G

def directed_to_undirected(G1):
    G = nx.Graph()
    for n in G1:
        if 'id' in G1.nodes[n]:
            G.add_node(n, id = G1.nodes[n]['id'], alias = G1.nodes[n]['alias'], weight=G1.nodes[n]['weight'], matched=G1.nodes[n]['matched'])
        else:
            print ("\n\nThis node has no id")
            print (n)
            print (G1.nodes[n])
            break
    done = set()
    elist = []
    for (n1, n2) in G1.edges():
        if (n1, n2) not in done:
            done.add((n1, n2))
            done.add((n2, n1))
            data = G1.get_edge_data(n1, n2)

            d = data['weight']
            wlist1 = copy.deepcopy(data['wlist'])
            #print ("wlist1",wlist1)
            etype1 = data['etype']

            if (n2, n1) in G1.edges():
                data1 = G1.get_edge_data(n2, n1)
                if data1['etype'] == 'Triple':
                    if data1['weight'] > d:  # Keeping maximum weight edge
                        d = data1['weight']
                    #print (len(data1['wlist']))
                    for w in data1['wlist']:
                        #print ("wlist",w,wlist1)
                        wlist1.append(w)
                    #print (len(wlist1))

            for i in range(0, len(wlist1)):
                if wlist1[i] > 1.0 and wlist1[i] <= 1.0001:
                    wlist1[i] = 1.0
            if d > 1.0 and d <= 1.0001:
                d = 1.0
            #elist.append((n1, n2, d, wlist1, etype1))
            #print (elist)
            G.add_edge(n1, n2, weight=d, wlist=wlist1, etype=etype1)

    flag = 0
    elist = sorted(elist, key=lambda x: x[2], reverse=True)

    for (n1, n2) in G.edges():
        data = G.get_edge_data(n1, n2)
        d = data['weight']
        wlist1 = data['wlist']

        if d > 1:
            flag += 1
        for ww in wlist1:
            if ww > 1:
                flag += 1
    #print("No. of neg weights ",flag)
    return G

def pred_match(labels, q_ent):
    matched = ''
    lab_stem = set()
    for label in labels:
        label = replace_symbols_in_relation(label)
        label_li = set(label.split())
        for lab in label_li:
            if lab not in stop_words:
                lab_stem |= set([lab, PS.stem(lab), LC.stem(lab), SB.stem(lab)])

    for terms in q_ent:
        if terms in stop_words: continue
        ter_stem = set()
        qterms = set(terms.split())
        for ter in qterms:
            if ter not in stop_words:
                ter_stem |= set([ter, PS.stem(ter), LC.stem(ter), SB.stem(ter)])

        if len(lab_stem.intersection(ter_stem)) > 0:
            matched = terms
            return matched
    return matched

def get_spo_for_build_graph(spo_lines,q_ent,hit_sta,sta_score,ent_sta):
    #print("\n\n terms: ")
    #print(q_ent)
    corner_ent = {}
    unique_SPO_dict = {}
    spo_fact = {}
    qkgspo = []

    for line in spo_lines:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue

        sta_id = triple[0]
        statement_id = sta_id.replace("-ps:", "").replace("-pq:", "")
        if statement_id in hit_sta: qkgspo.append(line)
        if statement_id not in spo_fact:
            spo_fact[statement_id] = {}
        n1_id = triple[1]
        n1_name = triple[2]
        n2 = triple[4]
        n3_id = triple[5]
        n3_name = triple[6]
        if n1_id not in ent_sta:
            ent_sta[n1_id] = []
        ent_sta[n1_id].append(statement_id)

        if n3_id not in ent_sta: ent_sta[n3_id] = []
        ent_sta[n3_id].append(statement_id)

        if sta_id.endswith("-ps:"):
            spo_fact[statement_id]["ps"] = (n1_id, n1_name, n2, n3_id, n3_name)

        if sta_id.endswith("-pq:"):
            if 'pq' not in spo_fact[statement_id]:
                spo_fact[statement_id]['pq'] = []
            spo_fact[statement_id]['pq'].append((n1_id, n1_name, n2, n3_id, n3_name))

    for sta in spo_fact:
        if 'ps' not in spo_fact[sta]: continue
        if sta not in hit_sta: continue
        ps = spo_fact[sta]['ps']
        n1_id = ps[0]
        n1_name = ps[1]
        n2 = ps[2] + "#" + sta
        n3_id = ps[3]
        n3_name = ps[4]
        score_n1 = 0.0
        score_n3 = 0.0
        for score in [sta_score[item] for item in ent_sta[n1_id]]:
            score_n1 += score
        for score in [sta_score[item] for item in ent_sta[n3_id]]:
            score_n3 += score
        matched_n1 = ''
        matched_n3 = ''
        if n1_id.startswith('corner#'):
            n1_id = n1_id.replace('corner#', '')
            n11 = n1_id.split('#')
            n1_id = n11[0]
            term = n11[2]

            term_words = set(term.split())
            for terms in q_ent:
                qterm = set(terms.split())
                if len(term_words.intersection(qterm)) > 0:
                    matched_n1 = terms
                    break

            if matched_n1 not in corner_ent:
                corner_ent[matched_n1] = set()
            corner_ent[matched_n1].add(n1_name + '::Entity::' + n1_id)

        if n3_id.startswith('corner#'):
            n3_id = n3_id.replace('corner#', '')
            n33 = n3_id.split('#')
            n3_id = n33[0]

            term = n33[2]
            term_words = set(term.split())

            for terms in q_ent:
                qterm = set(terms.split())
                if len(term_words.intersection(qterm)) > 0:
                    matched_n3 = terms
                    break

            if matched_n3 not in corner_ent:
                corner_ent[matched_n3] = set()
            corner_ent[matched_n3].add(n3_name + '::Entity::' + n3_id)

        if (n1_id, n2, n3_id) not in unique_SPO_dict:
            unique_SPO_dict[(n1_id, n2, n3_id)] = {}
        unique_SPO_dict[(n1_id, n2, n3_id)]['score_n1'] = score_n1
        unique_SPO_dict[(n1_id, n2, n3_id)]['score_n3'] = score_n3
        unique_SPO_dict[(n1_id, n2, n3_id)]['matched_n1'] = matched_n1
        unique_SPO_dict[(n1_id, n2, n3_id)]['matched_n3'] = matched_n3
        unique_SPO_dict[(n1_id, n2, n3_id)]['name_n1'] = n1_name
        unique_SPO_dict[(n1_id, n2, n3_id)]['name_n3'] = n3_name

        if 'pq' in spo_fact[sta]:
            pqstat = spo_fact[sta]['pq']
            for pq in pqstat:
                qn2 = pq[2] + "#" + sta
                qn3_id = pq[3]
                qn3_name = pq[4]
                score_qn3 = 0.0
                for score in [sta_score[item] for item in ent_sta[qn3_id]]:
                    score_qn3 += score
                matched_qn3 = ''

                if qn3_id.startswith('corner#'):
                    qn3_id = qn3_id.replace('corner#', '')
                    qn33 = qn3_id.split('#')
                    qn3_id = qn33[0]
                    term = qn33[2]
                    term_words = set(term.split())

                    for terms in q_ent:
                        qterm = set(terms.split())
                        if len(term_words.intersection(qterm)) > 0:
                            matched_qn3 = terms
                            break

                    if matched_qn3 not in corner_ent:
                        corner_ent[matched_qn3] = set()
                    corner_ent[matched_qn3].add(qn3_name + '::Entity::' + qn3_id)

                if 'qualifier' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'].append((qn2, qn3_id, qn3_name))

                if 'score_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'].append(score_qn3)

                if 'matched_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'].append(matched_qn3)

    return unique_SPO_dict, corner_ent, spo_fact, qkgspo

def call_main_GRAPH(instance, spo_list, q_ent, pro_info, seeds_paths, add_spo_line, add_spo_score):
    evidences = instance["candidate_evidences"]
    sta_score = {evidence["statement"]: evidence["score"] for evidence in evidences}
    hit_sta = [evidence["statement"] for evidence in evidences]
    #construct graph first time to check if the graph is connected
    ent_sta = {}
    unique_SPO_dict, corner_ent, spo_fact, qkgspo = get_spo_for_build_graph(spo_list, q_ent, hit_sta, sta_score, ent_sta)
    print("\nAdding SPO triple edges\n")
    G = build_graph_from_triple_edges(unique_SPO_dict, pro_info, sta_score)
    # change G from direct to undirect graph
    G = directed_to_undirected(G)

    del unique_SPO_dict
    #Add node weights
    seed_ids = set()
    for item in corner_ent:
        for e in corner_ent[item]:
            seed_ids.add(e.split("::")[2])

    number_of_components_original = nx.number_connected_components(G)
    number_of_components_connect = number_of_components_original
    if number_of_components_original > 1:
        S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        subg_group_seeds = []
        seed_pairs = []
        connect_sta = []
        for subg in S:
            subg_seed = []
            for node in subg:
                if node.split("::")[1] == "Entity":
                    qid = node.split("::")[2]
                    if qid in seed_ids:
                        subg_seed.append(qid)
            if len(subg_seed) > 0:
                subg_group_seeds.append(subg_seed)
            else:
                print("\n\nError !!! This subgraph has no seeds!!")

        if len(subg_group_seeds) > 1:
            for i in range(0, len(subg_group_seeds) - 1):
                for s1 in subg_group_seeds[i]:
                    for j in range(i + 1, len(subg_group_seeds)):
                        for s2 in subg_group_seeds[j]:
                            seed_pairs.append((s1, s2))

        print(f"seed pairs: {seed_pairs}.")
        for (s1, s2) in seed_pairs:
            str_li = [s1, s2]
            str_li.sort()
            str_pair = '||'.join(str_li)
            if str_pair in seeds_paths:
                print ("str_pair")
                print (str_pair)
                for item in seeds_paths[str_pair]:
                    for sta in item:
                        if len(sta) > 0 and sta not in connect_sta:
                            connect_sta.append(sta)


        ent_seed_score_map = {}
        for n1_id in ent_sta:
            if n1_id.startswith('corner#'):
                n1_id = n1_id.replace('corner#', '')
                n11 = n1_id.split('#')
                n1_id = n11[0]
                n1_score = n11[1]
                term = n11[2]
                ent_seed_score_map[n1_id] = {'score':n1_score,'term':term}


        adding_line_count = 0
        adding_lines = []
        for line in add_spo_line:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = triple[0].replace("-ps:", "").replace("-pq:", "")
            if statement_id in connect_sta:
                n1_id = triple[1]
                n3_id = triple[5]
                if n1_id.startswith('corner#'):
                    n1_id = n1_id.replace('corner#', '')
                    n1_id = 'corner#'+n1_id+'#'+ent_seed_score_map[n1_id]['score']+'#'+ent_seed_score_map[n1_id]['term']
                    line = line.replace(triple[1],n1_id)
                if n3_id.startswith('corner#'):
                    n3_id = n3_id.replace('corner#', '')
                    n3_id = 'corner#'+n3_id+'#'+ent_seed_score_map[n3_id]['score']+'#'+ent_seed_score_map[n3_id]['term']
                    line = line.replace(triple[5],n3_id)

                if line not in spo_list:
                    spo_list.append(line)
                    adding_lines.append(line)
                    adding_line_count += 1
        print("\n\nadding lines: " + str(adding_line_count))

        if len(connect_sta) > 0:
            for sta in connect_sta:

                if sta not in sta_score:
                    if sta not in add_spo_score:
                        print(add_spo_score)
                    sta_score[sta] = float(add_spo_score[sta])
            unique_SPO_dict, corner_ent, spo_fact, qkgspo = get_spo_for_build_graph(spo_list, q_ent, hit_sta + connect_sta, sta_score, ent_sta)
            G = build_graph_from_triple_edges(unique_SPO_dict, pro_info, sta_score)
            # change G from direct to undirect graph
            G = directed_to_undirected(G)
            del spo_fact
            del unique_SPO_dict
            seed_ids = set()
            for item in corner_ent:
                for e in corner_ent[item]:
                    seed_ids.add(e.split("::")[2])

            number_of_components_connect = nx.number_connected_components(G)
            print("\n\nnumber_of_components", number_of_components_original)
            print("\n\nnumber_of_components_connect", number_of_components_connect)


    for n1 in G:
        nn1 = n1.split('::')
        id = G.nodes[n1]['id']
        label = nn1[0]
        alias = G.nodes[n1]['alias'].copy()
        alias.append(label)
        if nn1[1] == 'Predicate':
            G.nodes[n1]['weight'] = sta_score[nn1[2]]
            G.nodes[n1]['matched'] = pred_match(alias, q_ent)

    print("\n\nGetting cornerstones \n\n")
    corner2 = {}
    for n in G:
        id = G.nodes[n]['id']
        if G.nodes[n]['matched'] != '':
            if G.nodes[n]['matched'] not in corner2:
                corner2[G.nodes[n]['matched']] = []
            corner2[G.nodes[n]['matched']].append(n)
        elif id in seed_ids:
            print (n)
            if G.nodes[n]['matched'] not in corner2:
                corner2[G.nodes[n]['matched']] = []
            corner2[G.nodes[n]['matched']].append(n)

    print ("Final query terms -->", len(q_ent), q_ent)
    print ('seed nodes -->', seed_ids)
    print ('corner entities -->', corner_ent)
    cornerstone = {}
    for v in corner2:
        # print("v in corner  -->")
        # print(v)
        for e in corner2[v]:
            #print(e)
            cornerstone[e] = v

    instance["cornerstone"] = cornerstone

    return G, cornerstone, qkgspo, number_of_components_original, number_of_components_connect

def myHandler(signum, frame):
    print("time out!!!")
    exit()

def remove_hanging_nodes_from_gst(unionGST, cornerstone):
    unionGST_pruned = unionGST.copy()
    for n in unionGST.nodes():
        nn = n.split('::')
        if nn[1] == 'Predicate' and n not in cornerstone and unionGST.degree(n) == 1:
            unionGST_pruned.remove_node(n)
    cornerstone_nodes = [node for node in unionGST_pruned.nodes() if node in cornerstone]
    #print("\nlength of cornerstone in pruned GSTs: ", len(cornerstone_nodes))
    if len(unionGST_pruned.nodes()) > len(cornerstone_nodes):
        if 1 not in [unionGST_pruned.degree(n) for n in unionGST_pruned if n not in cornerstone and n.split('::')[1] == 'Predicate']:
            #print ("\nlength of pruned GSTs: ", len(unionGST_pruned.nodes()))
            return unionGST_pruned
        else:
            return remove_hanging_nodes_from_gst(unionGST_pruned, cornerstone)
    else:
        return unionGST_pruned

class QuestionUnionGST():
    def __init__(self, config, property):
        self.config = config
        self.logger = get_logger(__name__, self.config)
        self.topf = self.config["fs_max_evidences"]
        self.topg = self.config["top-gst-number"]
        self.pro_info = property
        # initialize clocq for KB-facts
        if self.config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

    def get_completedgst_spo(self, spo_list, completedGST, proinfo):
        completedgstspo = []
        state = {}
        for line in spo_list:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            ps_rel = triple[3]
            pq_rel = triple[4]
            statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "")
            ps_rel_type, ps_rel_label, ps_rel_alias = property_type_label_alias(ps_rel, proinfo)
            pq_rel_type, pq_rel_label, pq_rel_alias = property_type_label_alias(pq_rel, proinfo)
            if statement_id not in state:
                state[statement_id] = dict()
                state[statement_id]['ps'] = {}
                state[statement_id]['pq'] = {}
            if "-ps:" in line.split("||")[0]:
                state[statement_id]['ps'].update({ps_rel_label: line})
            if "-pq:" in line.split("||")[0]:
                if pq_rel_label not in state[statement_id]['pq'].keys():
                    state[statement_id]['pq'][pq_rel_label] = []
                state[statement_id]['pq'][pq_rel_label].append(line)

        completedgstspo_sort_sta = {}
        for node in [n for n in completedGST.nodes() if n.split('::')[1] == 'Predicate']:
            statement_id = node.split("::")[2]
            predicate_label = node.split("::")[0]
            if statement_id not in completedgstspo_sort_sta:
                completedgstspo_sort_sta[statement_id] = dict()
                completedgstspo_sort_sta[statement_id]['ps'] = ''
                completedgstspo_sort_sta[statement_id]['pq'] = []
            if predicate_label in state[statement_id]['ps']:
                completedgstspo.append(state[statement_id]['ps'][predicate_label])
                completedgstspo_sort_sta[statement_id]['ps'] = state[statement_id]['ps'][predicate_label]
            if predicate_label in state[statement_id]['pq']:
                pq_lines = state[statement_id]['pq'][predicate_label]
                #print (pq_lines)
                pq_entity = [nb.split("::")[2] for nb in completedGST.neighbors(node) if nb.split("::")[1] == 'Entity'][0]
                for line in pq_lines:
                    triple = line.strip().split('||')
                    #n1_id = triple[1]
                    n3_id = triple[5]
                    if 'corner#' in n3_id:
                        n3_id = n3_id.split("#")[1]
                    if pq_entity == n3_id:
                        completedgstspo.append(line)
                        completedgstspo_sort_sta[statement_id]['pq'].append(line)

        return completedgstspo

    def get_completedgst_graph(self, QKG, unionGST, unionGST_prune, cornerstone):
        completedgst = nx.Graph()
        isolate_nodes = [n for n in unionGST_prune.nodes() if n.split('::')[1] == 'Entity' and unionGST_prune.degree(n) == 0]

        if len(isolate_nodes) == len(unionGST_prune.nodes()):
            self.logger.info(f"len(isolate_nodes): {len(isolate_nodes)}.")
            completedgst = unionGST.copy()
            for n in unionGST:
                if n.split('::')[1] == 'Predicate' and n in cornerstone:
                    for nb in QKG.neighbors(n):
                        completedgst.add_node(nb)
                        completedgst.add_edge(n, nb)
                        if nb.split('::')[1] == 'Predicate':
                            for nnb in QKG.neighbors(nb):
                                if nnb.split('::')[1] == 'Entity':
                                    completedgst.add_node(nnb)
                                    completedgst.add_edge(nb, nnb)

                if n.split('::')[1] == 'Predicate' and n not in cornerstone:
                    entity_neighbor = [nb for nb in QKG.neighbors(n) if nb.split('::')[1] == 'Entity']
                    pred_neighbor = [nb for nb in QKG.neighbors(n) if nb.split('::')[1] == 'Predicate']
                    # main predicate has two entities as neighbors
                    if len(entity_neighbor) == 2:
                        # add entity node into completed gst graph
                        for en_node in entity_neighbor:
                            completedgst.add_node(en_node)
                            completedgst.add_edge(n, en_node)
                    # qualifier predicate has one entity as neighbor
                    elif len(entity_neighbor) == 1 and len(pred_neighbor) >= 1:
                        en_node = entity_neighbor[0]
                        pred_node = pred_neighbor[0]
                        completedgst.add_node(en_node)
                        completedgst.add_edge(n, en_node)
                        completedgst.add_node(pred_node)
                        completedgst.add_edge(n, pred_node)
                        for nnb in QKG.neighbors(pred_node):
                            if nnb.split('::')[1] == 'Entity':
                                completedgst.add_node(nnb)
                                completedgst.add_edge(pred_node, nnb)

        else:
            for n in unionGST_prune.nodes():
                if n.split('::')[1] == 'Entity' and unionGST_prune.degree(n) == 0:
                    continue
                completedgst.add_node(n)

            for (n1, n2) in unionGST_prune.edges():
                if n1 in completedgst and n2 in completedgst:
                    completedgst.add_edge(n1, n2)

            for n in unionGST_prune:
                if n.split('::')[1] == 'Predicate' and n in cornerstone:
                    for nb in QKG.neighbors(n):
                        completedgst.add_node(nb)
                        completedgst.add_edge(n, nb)
                        if nb.split('::')[1] == 'Predicate':
                            for nnb in QKG.neighbors(nb):
                                if nnb.split('::')[1] == 'Entity':
                                    completedgst.add_node(nnb)
                                    completedgst.add_edge(nb, nnb)

                if n.split('::')[1] == 'Predicate' and n not in cornerstone:
                    entity_neighbor = [nb for nb in QKG.neighbors(n) if nb.split('::')[1] == 'Entity']
                    pred_neighbor = [nb for nb in QKG.neighbors(n) if nb.split('::')[1] == 'Predicate']
                    # main predicate has two entities as neighbors
                    if len(entity_neighbor) == 2:
                        # add entity node into completed gst graph
                        for en_node in entity_neighbor:
                            completedgst.add_node(en_node)
                            completedgst.add_edge(n, en_node)
                    # qualifier predicate has one entity as neighbor
                    elif len(entity_neighbor) == 1:
                        en_node = entity_neighbor[0]
                        pred_node = pred_neighbor[0]
                        completedgst.add_node(en_node)
                        completedgst.add_edge(n, en_node)
                        completedgst.add_node(pred_node)
                        completedgst.add_edge(n, pred_node)
                        for nnb in QKG.neighbors(pred_node):
                            if nnb.split('::')[1] == 'Entity':
                                completedgst.add_node(nnb)
                                completedgst.add_edge(pred_node, nnb)

        for (n1, n2) in QKG.edges():
            if n1 in completedgst and n2 in completedgst:
                completedgst.add_edge(n1, n2)
        return completedgst

    def get_connectivity_info(self, instance):
        seeds_paths = {}
        score = {}
        add_spo_score = {}
        result = instance["connectivity"]

        seeds_paths.update(result["best_paths"])
        score.update(result['score'])
        add_spo_line = result["spo_line"].copy()

        for pair in seeds_paths:
            sta_in_paths = seeds_paths[pair]
            path_score = score[pair]
            for item in sta_in_paths:
                for sta in item:
                    add_spo_score.update({sta: float(path_score)})
        return seeds_paths, add_spo_line, add_spo_score

    def get_entities_from_spo_list(self, spo_list):
        spo_entity = list()
        for line in spo_list:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            sub_id = triple[1]
            sub_label = triple[2]
            obj_id = triple[5]
            obj_label = triple[6]

            if 'corner#' in sub_id:
                sub_id = sub_id.replace('corner#', '').split('#')[0]
            if 'corner#' in obj_id:
                obj_id = obj_id.replace('corner#', '').split('#')[0]

            if {"id":sub_id, "label":sub_label} not in spo_entity:
                spo_entity.append({"id":sub_id, "label":sub_label})
            if {"id": obj_id, "label": obj_label} not in spo_entity:
                spo_entity.append({"id": obj_id, "label": obj_label})
        return spo_entity

    def get_gst_for_instance(self, instance):
        output_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        nerd = self.config["nerd"]
        question_id = instance["Id"]
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        output_dir = os.path.join(output_dir, nerd, f"ers-{fs_max_evidences}-gst-{topg}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        gst_output_path = os.path.join(output_dir, f"{question_id}_gst.gexf")
        complete_gst_output_path = os.path.join(output_dir, f"{question_id}_comgst.gexf")
        spo_list = list()
        seeds_paths, add_spo_line, add_spo_score = self.get_connectivity_info(instance)
        question = instance["Question"]
        evidences = instance["candidate_evidences"]
        for evidence in evidences:
            spo_list += evidence["statement_spo_list"]
        qterms = replace_symbols_in_question(question)
        q_ent = set()
        for term in qterms.split():
            if term not in stop_words:
                q_ent.add(term.lower())
        self.logger.info(f"question entities: {q_ent}.")
        t1 = time.time()

        QKG, cornerstone, qkgspo, number_of_components, number_of_components_connect = call_main_GRAPH(instance, spo_list, q_ent, self.pro_info, seeds_paths, add_spo_line, add_spo_score)
        entity_weight = {QKG.nodes[n]['id']: QKG.nodes[n]['weight'] for n in QKG}
        S = [QKG.subgraph(c).copy() for c in nx.connected_components(QKG)]
        GST_list = []
        for subg in S:
            try:
                signal.signal(signal.SIGALRM, myHandler)
                signal.alarm(MAX_TIME)
                GST = call_main_rawGST(subg, cornerstone, self.topg)
                print("DONE GST Algorithm...", str(question_id))
                if GST:
                    GST_list.append(GST)
            except:
                self.logger.info(f"Time out: {question}.")
                continue

            t2 = time.time()
            self.logger.info(f"Running time for one subgraph: {t2-t1} seconds.")

        if len(GST_list) > 0:
            unionGST = nx.compose_all(GST_list)
            unionGST_prune = remove_hanging_nodes_from_gst(unionGST, cornerstone)
            completedGST = self.get_completedgst_graph(QKG, unionGST, unionGST_prune, cornerstone)
            completedSPO = self.get_completedgst_spo(spo_list, completedGST, self.pro_info)
            completedgst_can_entities = [{"id": node.split("::")[2], "label":node.split("::")[0]} for node in completedGST.nodes() if node.split("::")[1] == "Entity"]
            nx.write_gexf(unionGST, gst_output_path)
            nx.write_gexf(completedGST, complete_gst_output_path)
            return entity_weight, completedSPO, cornerstone, completedgst_can_entities
        else:
            return entity_weight, [], {}, []