import os
import json
import random
import csv
import globals
import truecase
import re
import networkx as nx
import copy
#import pickle

from get_CLOCQ_Wikidata_SPOs import get_wikidata_twohoptuplesfromclocq

cfg = globals.get_config(globals.config_file)
dev = cfg["data_path"] + cfg["dev_data"]
train = cfg["data_path"] + cfg["train_data"]
DATASETs = [dev, train]
pro_info = globals.ReadProperty.init_from_config().property
TRAINING_FILE = cfg["data_path"] + "phase1_relevant_fact_selection_trainset_new.csv"
train_list = []
#ques_answer_1hop = cfg["data_path"] + "ques_answer_1hop.pkl"
#ques_answer_2hop = cfg["data_path"] + "ques_answer_2hop.pkl"

def _get_answer_coverage(GT, all_entities):

    if len(GT) == 0:
        return -1.
    elif len(all_entities) == 0:
        return -1.
    else:
        found, total = 0., 0
        for answer in GT:
            if answer.lower() in all_entities: found += 1.
            total += 1
        return found / total

def get_extractentities_spo(spos):
    spo_entity = list()
    # # print(spo_file)
    # if not os.path.exists(spo_file):
    #     print(spo_file + ' not found!')
    #     return spo_entity
    # f22 = open(spo_file, 'r')
    # spos = f22.readlines()
    # f22.close()
    for line in spos:
        # li = line.split("|")
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        # sta_id = triple[0]
        sub = triple[1]
        # n1_name = triple[2]
        # n2 = triple[4]
        obj = triple[5]
        # n3_name = triple[6]
        if 'T00:00:00Z' in sub:
            sub = sub.replace('T00:00:00Z', '')
        if 'T00:00:00Z' in obj:
            obj = obj.replace('T00:00:00Z', '')
        # obj_na = line.split("|")[6]
        if 'corner#' in sub:
            sub = sub.replace('corner#', '').split('#')[0]
        if 'corner#' in obj:
            obj = obj.replace('corner#', '').split('#')[0]
        spo_entity.append(sub.lower())
        spo_entity.append(obj.lower())
    return list(set(spo_entity))

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

def build_graph_from_triple_edges(unique_SPO_dict, proinfo):
    G = nx.DiGraph()
    pred_count = {}
    for (n1, n2, n3) in unique_SPO_dict:
        n2_id = n2.split("#")[0]
        n2_sta = n2.split("#")[1]
        n22t, n22l, n22a = property_type_label_alias(n2_id, proinfo)
        n1_name = unique_SPO_dict[(n1, n2, n3)]['name_n1']
        n3_name = unique_SPO_dict[(n1, n2, n3)]['name_n3'].replace('T00:00:00Z','')
        n11 = n1_name + "::Entity::" + n1
        if n22l not in pred_count:
            pred_count[n22l] = 1
        else:
            pred_count[n22l] = pred_count[n22l] + 1
        n22 = n22l + "::Predicate::" + n2_sta + "::" + str (pred_count[n22l])
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
        sta_weight = 0.0
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
                qn22t, qn22l, qn22a = property_type_label_alias (qn2_id, proinfo)
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

def get_graph(spo_file, two_hop_lines, proinfo):
    unique_SPO_dict = {}
    ent_sta = {}
    spo_fact = {}
    remove_duplicated_line = []
    with open(spo_file, 'r', encoding='utf-8') as f11:
        for line in f11:
            if line not in remove_duplicated_line:
                remove_duplicated_line.append(line)
    for line in two_hop_lines:
        if line not in remove_duplicated_line:
            remove_duplicated_line.append(line)

    #with open (spo_file, 'r', encoding = 'utf-8') as f11:
    for line in remove_duplicated_line:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        sta_id = triple[0]
        statement_id = sta_id.replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id not in spo_fact:
            spo_fact[statement_id] = {}
        n1_id = triple[1]
        n1_name = triple[2]
        n2 = triple[4]
        n3_id = triple[5]
        n3_name = triple[6]

        if n1_id not in ent_sta: ent_sta[n1_id] = []
        ent_sta[n1_id].append(statement_id)

        if n3_id not in ent_sta: ent_sta[n3_id] = []
        ent_sta[n3_id].append(statement_id)

        if sta_id.endswith("-ps:"):
            spo_fact[statement_id]["ps"] = (n1_id,n1_name,n2,n3_id,n3_name)

        if sta_id.endswith("-pq:"):
            if 'pq' not in spo_fact[statement_id]:
                spo_fact[statement_id]['pq'] = []
            spo_fact[statement_id]['pq'].append((n1_id, n1_name,n2,n3_id, n3_name))

    for sta in spo_fact:
        if 'ps' not in spo_fact[sta]: continue
        ps = spo_fact[sta]['ps']
        n1_id = ps[0]
        n1_name = ps[1]
        n2 = ps[2] + "#" + sta
        n3_id = ps[3]
        n3_name = ps[4]
        score_n1 = 0.0
        score_n3 = 0.0
        matched_n1 = ''
        matched_n3 = ''
        if n1_id.startswith('corner#'):
            n1_id = n1_id.replace('corner#', '')
            n11 = n1_id.split('#')
            n1_id = n11[0]

        if n3_id.startswith('corner#'):
            n3_id = n3_id.replace('corner#', '')
            n33 = n3_id.split('#')
            n3_id = n33[0]

        if (n1_id,n2,n3_id) not in unique_SPO_dict:
            unique_SPO_dict[(n1_id,n2,n3_id)] = {}
        unique_SPO_dict[(n1_id,n2,n3_id)]['score_n1'] = score_n1
        unique_SPO_dict[(n1_id,n2,n3_id)]['score_n3'] = score_n3
        unique_SPO_dict[(n1_id,n2,n3_id)]['matched_n1'] = matched_n1
        unique_SPO_dict[(n1_id,n2,n3_id)]['matched_n3'] = matched_n3
        unique_SPO_dict[(n1_id,n2,n3_id)]['name_n1'] = n1_name
        unique_SPO_dict[(n1_id,n2,n3_id)]['name_n3'] = n3_name

        if 'pq' in spo_fact[sta]:
            pqstat = spo_fact[sta]['pq']
            for pq in pqstat:
                qn2 = pq[2] + "#" + sta
                qn3_id = pq[3]
                qn3_name = pq[4]
                score_qn3 = 0.0
                matched_qn3 = ''

                if qn3_id.startswith('corner#'):
                    qn3_id = qn3_id.replace('corner#', '')
                    qn33 = qn3_id.split('#')
                    qn3_id = qn33[0]

                if 'qualifier' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['qualifier'].append((qn2,qn3_id,qn3_name))

                if 'score_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['score_qn3'].append(score_qn3)

                if 'matched_qn3' not in unique_SPO_dict[(n1_id, n2, n3_id)]:
                    unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'] = []
                unique_SPO_dict[(n1_id, n2, n3_id)]['matched_qn3'].append(matched_qn3)

    del spo_fact
    #print("\nAdding SPO triple edges\n")

    #Question vectors and node vectors are built inside
    G = build_graph_from_triple_edges(unique_SPO_dict, proinfo)
    del unique_SPO_dict

    # change G from direct to undirect graph
    #print("\n\nChanging directed graph to undirected graph\n\n")
    G = directed_to_undirected(G)
    #Add node weight
    return G

def get_entities_spo(spo_file):
    spo_entity = list()
    if not os.path.exists(spo_file):
        print(spo_file + ' not found!')
        return spo_entity
    f22 = open(spo_file, 'r')
    spos = f22.readlines()
    f22.close()
    for line in spos:
        triple = line.strip().split("||")
        sub = triple[1]
        obj = triple[5]
        if "corner#" in sub: sub = sub.split("#")[1]
        if "corner#" in obj: obj = obj.split("#")[1]
        if 'T00:00:00Z' in sub:
            sub = sub.replace('T00:00:00Z', '')
        if 'T00:00:00Z' in obj:
            obj = obj.replace('T00:00:00Z', '')
        spo_entity.append(sub)
        spo_entity.append(obj)
    return list(set(spo_entity))

def write_2hopspo_to_lines(fact_dic, have_answer):
    lines = []
    # for entities
    ENT_PATTERN = re.compile('^Q[0-9]+$')
    # for predicates
    PRE_PATTERN = re.compile('^P[0-9]+$')
    for sta in fact_dic:
        if sta not in have_answer: continue
        t = fact_dic[sta]['ps'][0]
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

        p = sub + "||" + subname + "||" + pre
        ps_line = sta + "-ps:" + "||" + p + "||" + pre + "||" + obj + "||" + str(objname) + '\n'
        lines.append(ps_line)
        for pqt in fact_dic[sta]['pq']:
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
            pq_line = sta + "-pq:" + "||" + p + "||" + pre + "||" + obj + "||" + objname + '\n'
            if pq_line not in lines:
                lines.append(pq_line)

    return lines

def _read_seeds(tagme, elq):
    """Return map from question ids to seed entities."""
    seeds = set()

    with open(tagme) as f:
        for line in f:
            entity, score, text = line.strip().split('\t')
            seeds.add(entity)

    with open(elq) as f:
        for line in f:
            entity, score, text = line.strip().split('\t')
            seeds.add(entity)

    return list(seeds)

def get_intermedia_entity(id_path, spo_file, pro_info, GT):
    tagme_wiki_ids_file = id_path + '/' + 'wiki_ids_tagme.txt'
    elq_wiki_ids_file = id_path + '/' + 'wiki_ids_elq.txt'
    new_answers = set()
    wiki_ids = set()
    seeds = _read_seeds(tagme_wiki_ids_file, elq_wiki_ids_file)
    #spo_2hop_file = id_path + '/' + 'SPO_elq_tagme_2hop.txt'
    # ENT_PATTERN = re.compile('^Q[0-9]+$')
    # print("\n\nGenerating 2 hop facts from SPOS...\n")
    # entities = get_entities_spo(spo_file)
    # qids = [item for item in entities if ENT_PATTERN.match(item) != None]
    # candidates = list(set(qids).difference(set(seeds)))
    # print ("\n\n length of candidates: ", str(len(candidates)))
    fact_dic = get_wikidata_twohoptuplesfromclocq(wiki_ids)
    have_answer = []
    for sta in fact_dic:
        flag = 0
        t = fact_dic[sta]['ps'][0]
        sub = t[0]['id'].strip('"')
        obj = t[2]['id'].strip('"')
        if 'T00:00:00Z' in sub:
            sub = sub.replace('T00:00:00Z', '')
        if 'T00:00:00Z' in obj:
            obj = obj.replace('T00:00:00Z', '')
        if sub.lower() in GT or obj.lower() in GT:
            flag = 1
        for pqt in fact_dic[sta]['pq']:
            pq_obj = pqt[1]['id'].strip('"')
            pq_obj = pq_obj.replace('T00:00:00Z', '')
            if pq_obj.lower() in GT: flag = 1
        if flag == 1: have_answer.append(sta)
    two_hop_lines = write_2hopspo_to_lines(fact_dic, have_answer)
    #print("\n\n2 hop facts SPOs generated...")
    two_hop_entities = get_extractentities_spo(two_hop_lines)
    print("length of 2 hop entity: ", str(len(two_hop_entities)))
    spo_2hop_ac = _get_answer_coverage(GT, two_hop_entities)
    if spo_2hop_ac <= 0:
        return new_answers
    print("\n\nanswer in 2 hop facts!")
    G = get_graph(spo_file, two_hop_lines, pro_info)
    start = []
    end = []
    for n in G.nodes():
        id = n.split('::')[2]
        if id in seeds: start.append(n)
        if id.replace("T00:00:00Z","").lower() in GT: end.append(n)

    for e in end:
        path_list = []
        for s in start:
            if s != e:
                try:
                    path = nx.shortest_path(G, s, e)
                except:
                    print ("\nno path between " + s + " and " + e)
                    continue
                if (len(path)) > 0:
                    path_list.append({"path": path, "len": len(path)})
        path_list.sort(key=lambda k : k['len'])
        print (path_list[0])
        for item in path_list[0]["path"]:
            if item.split('::')[1] == "Entity" and item not in s and item not in e:
                new_answers.add(item.split('::')[2].lower())

    return new_answers

count = 0
answer_in_1hop = []
answer_in_2hop = []
for DATASET in DATASETs:
    data = json.load(open(DATASET))
    for item in data:
        GT = []
        positive = []
        pos_rel = []
        negative = []
        postive_train = []
        negative_train = []
        ques_id = str(item['Id'])
        text = item['Question']
        answers = item['Answer']
        for ans in answers:
            if 'WikidataQid' in ans:
                Qid = ans['WikidataQid'].lower()
                GT.append(Qid)
            if "AnswerArgument" in ans:
                GT.append(ans['AnswerArgument'].replace('T00:00:00Z', '').lower())

        id_path = cfg['ques_path'] + 'ques_' + ques_id
        spo_file = id_path + '/SPO_new.txt'
        spo_sta = {}
        ans_sta = {}
        main_rel_sta = {}
        if not (os.path.exists(spo_file)):
            print ("spo_file not exist!")
            continue
        # file_entities = get_extractentities_spo(spo_file)
        # spo_ac = _get_answer_coverage(GT, file_entities)
        # if spo_ac > 0:
        #     answer_in_1hop.append(ques_id)
        #     print("\nanswer in one hop facts! " + str(ques_id))
        # elif spo_ac == -1: continue
        # elif spo_ac == 0:
        #     new_ans = get_intermedia_entity(id_path, spo_file, pro_info, GT)
        #     if len(new_ans) >0:
        #         GT += list(new_ans)
        #         answer_in_2hop.append(ques_id)
        #         print("\nanswer in two hop facts! " + str(ques_id))

        with open(spo_file) as f:
            for line in f:
                triple = line.strip().split('||')
                if len(triple) < 7 or len(triple) > 7: continue
                statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
                sub = triple[1].replace('T00:00:00Z', '')
                obj = triple[5].replace('T00:00:00Z', '')
                if 'corner#' in sub: sub = sub.replace('corner#', '').split('#')[0]
                if 'corner#' in obj: obj = obj.replace('corner#', '').split('#')[0]
                sub_name = triple[2].replace('T00:00:00Z', '')
                obj_name = triple[6].replace('T00:00:00Z', '')
                rel_name = triple[4]
                main_rel_name = triple[3]
                if triple[4] in pro_info:
                    rel_name = pro_info[triple[4]]['label']
                if triple[3] in pro_info:
                    main_rel_name = pro_info[triple[3]]['label']
                if statement_id not in spo_sta:
                    spo_sta[statement_id] = dict()
                    spo_sta[statement_id]['ps'] = []
                    spo_sta[statement_id]['pq'] = []
                if statement_id not in ans_sta:
                    ans_sta[statement_id] = []
                ans_sta[statement_id].append(sub.lower())
                ans_sta[statement_id].append(obj.lower())
                main_rel_sta[statement_id] = main_rel_name

                if "-ps:" in line.split("||")[0]:
                    spo_sta[statement_id]['ps'].append(sub_name + ' ' + rel_name + ' ' + obj_name)
                if "-pq:" in line.split("||")[0]:
                    spo_sta[statement_id]['pq'].append(' ' + rel_name + ' ' + obj_name)

        for statement_id in ans_sta:
            pos_flag = 0
            for ent in ans_sta[statement_id]:
                if ent in GT:
                    pos_flag = 1
                    break
            if pos_flag == 1:
                positive.append(statement_id)
                pos_rel.append(main_rel_sta[statement_id])

        for statement_id in ans_sta:
            pos_flag = 0
            for ent in ans_sta[statement_id]:
                if len(GT) > 0  and ent in GT:
                    pos_flag = 1
                    break
            if pos_flag == 0 and main_rel_sta[statement_id] not in pos_rel:
                negative.append(statement_id)

        if len(positive) > 0 and len(negative) > 0:
            count += 1
            positive_train = random.sample(positive, 1)
            if len(negative) < 5:
                negative_train = random.sample(negative, len(negative))
            else:
                negative_train = random.sample(negative, 5)
            train_id = 'train_' + ques_id
            train_ques_text = text
            train_ques_text = truecase.get_true_case(train_ques_text)
            context = " ".join(spo_sta[positive_train[0]]['ps']) + " and".join(spo_sta[positive_train[0]]['pq'])
            train_list.append([train_id,train_ques_text,context,1])
            for neg_sta in negative_train:
                context = (" ").join(spo_sta[neg_sta]['ps']) + (" and").join(spo_sta[neg_sta]['pq'])
                if [train_id,train_ques_text,context,0] not in train_list:
                    train_list.append([train_id,train_ques_text,context,0])

with open(TRAINING_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["questionId", "question", "context", "label"])
    for sample in train_list:
        #print (sample)
        writer.writerow(sample)

#pickle.dump(answer_in_1hop, open(ques_answer_1hop, 'wb'))
#pickle.dump(answer_in_2hop, open(ques_answer_2hop, 'wb'))
print("total of questions having positive samples", str(count))
#print("total of questions having positive samples in 1-hop", str(len(answer_in_1hop)))
#print("total of questions having positive samples in 2-hop", str(len(answer_in_2hop)))

