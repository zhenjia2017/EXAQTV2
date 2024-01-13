"""Script to get relational graph from answer_graph (temporal enhanced completed GSTs).
"""
import json
import networkx as nx
from tqdm import tqdm
import time
import re
from exaqt.evaluation import answer_presence, answer_presence_gst

NUM_PATTERN = re.compile('^\+[0-9]+')
def get_answer(answers):
    """extract unique answers from dataset."""
    GT = list()
    # print (answers)
    for answer in answers:
        if "T00:00:00Z" in answer["id"]:
            answer["id"] = answer["id"].replace("T00:00:00Z", "")
        GT.append((answer["id"], answer['label'].lower()))
    return GT

def get_triple_from_SPO(facts, enhance_facts, pro_info):
    #get spo triples like graftnet
    #s:subject
    #o:intermediate if there is pq statement

    tuples = []
    date_in_f = dict()
    state = dict()
    t_rels = set()
    t_ens = set()
    ens =set()
    rels = set()
    tempstate = dict()
    for line in enhance_facts:
        triple = line.strip().split('||')
        statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id not in state:
            state[statement_id] = {'ps':[],'pq':[]}
        if statement_id not in tempstate:
            tempstate[statement_id] = {'ps_spo':[], 'pq_spo':set(), 'date':set()}
        sub_id = triple[1]
        obj_id = triple[5]
        if 'corner#' in sub_id: sub_id = sub_id.replace('corner#', '').split('#')[0]
        if 'corner#' in obj_id: obj_id = obj_id.replace('corner#', '').split('#')[0]
        if NUM_PATTERN.match(obj_id):
            obj_id = obj_id.lstrip("+")
        if NUM_PATTERN.match(sub_id):
            sub_id = sub_id.lstrip("+")
        sub_name = triple[2].replace('T00:00:00Z', '')
        obj_name = triple[6].replace('T00:00:00Z', '')
        rel = triple[4]
        rel_name = rel
        if rel in pro_info:
            rel_name = pro_info[rel]['label']
            if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time":
                obj_id = obj_id.replace('T00:00:00Z', '')
                tempstate[statement_id]['date'].add((rel_name, obj_id))

            elif "T00:00:00Z" in obj_id:
                obj_id = obj_id.replace('T00:00:00Z', '')
                tempstate[statement_id]['date'].add((rel_name, obj_id))
        if "-ps:" in triple[0]:
            #ps_spo = {'sub_id':sub_id, 'sub_name': sub_name, 'rel': rel_name, 'obj_id':obj_id, 'obj_name': obj_name}
            ps_spo = (sub_id, sub_name, rel_name, obj_id, obj_name)
            tempstate[statement_id]['ps_spo'].append(ps_spo)
            state[statement_id]['ps'].append(line)
        if "-pq:" in triple[0] and line not in state[statement_id]['pq']:
            state[statement_id]['pq'].append(line)
            pq_spo = (sub_id, sub_name, rel_name, obj_id, obj_name)
            tempstate[statement_id]['pq_spo'].add(pq_spo)

    tkgfacts = get_qtkg(tempstate)

    for line in facts:
        triple = line.strip().split('||')
        statement_id = triple[0].replace("-ps:", "").replace("-pq:", "").lower()
        if statement_id not in state:
            state[statement_id] = dict()
            state[statement_id]['ps'] = []
            state[statement_id]['pq'] = []
        if "-ps:" in triple[0]:
            state[statement_id]['ps'].append(line)
        if "-pq:" in triple[0] and line not in state[statement_id]['pq']:
            state[statement_id]['pq'].append(line)

    for statement_id in state:
        if statement_id not in date_in_f:
            date_in_f[statement_id] = dict()
            date_in_f[statement_id]['tuple'] = []
            date_in_f[statement_id]['date'] = []
        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) == 0:
            ps_lines = state[statement_id]['ps']
            for line in ps_lines:
                triple = line.strip().split("||")
                sub = triple[1]
                sub_name = triple[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                if NUM_PATTERN.match(sub):
                    sub = sub.lstrip("+")
                rel = triple[3]
                rel_name = rel
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    #if pro_info[rel]["type"] == "http://wikiba.se/ontology#ExternalId" or pro_info[rel]["type"] == "http://wikiba.se/ontology#Url" or pro_info[rel]["type"] == "http://wikiba.se/ontology#CommonsMedia": continue
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                        t_ens.add(obj)
                if "T00:00:00Z" in obj and not obj.startswith("_:"):
                    obj = obj.replace("T00:00:00Z", "")
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                    t_ens.add(obj)
                    t_rels.add(rel_name)

                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": sub, "text": sub_name},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": obj, "text": obj_name}])

                ens.add(sub)
                ens.add(obj)
                rels.add(rel_name)

        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) > 0:
            ps_lines = state[statement_id]['ps']
            pq_lines = state[statement_id]['pq']
            for line in ps_lines:
                triple = line.strip().split("||")
                sub = triple[1]
                sub_name = triple[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                if NUM_PATTERN.match(sub):
                    sub = sub.lstrip("+")
                rel = triple[3]
                rel_name = rel
                if rel in pro_info:
                    #if pro_info[rel]["type"] == "http://wikiba.se/ontology#ExternalId" or pro_info[rel]["type"] == "http://wikiba.se/ontology#Url" or pro_info[rel]["type"] == "http://wikiba.se/ontology#CommonsMedia": continue
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                        t_ens.add(obj)

                if "T00:00:00Z" in obj and not obj.startswith("_:"):
                    obj = obj.replace("T00:00:00Z","")
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_name})
                    t_ens.add(obj)
                    t_rels.add(rel_name)

                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": sub, "text": sub_name},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": statement_id, "text": statement_id}])
                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": statement_id, "text": statement_id},
                    {"rel_id": rel, "text": rel_name},
                    {"kb_id": obj, "text": obj_name}])

                ens.add(sub)
                ens.add(statement_id)
                ens.add(obj)
                rels.add(rel_name)

            for line in pq_lines:
                triple = line.strip().split("||")
                obj = triple[5]
                obj_name = triple[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                rel_ps = triple[3]
                rel_pq = triple[4]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                #if rel_ps in pro_info:
                #    if pro_info[rel_ps]["type"] == "http://wikiba.se/ontology#ExternalId" or pro_info[rel_ps]["type"] == "http://wikiba.se/ontology#Url" or pro_info[rel_ps]["type"] == "http://wikiba.se/ontology#CommonsMedia": continue
                rel_pq_name = rel_pq
                if rel_pq in pro_info:
                    #if pro_info[rel_pq]["type"] == "http://wikiba.se/ontology#ExternalId" or pro_info[rel_pq]["type"] == "http://wikiba.se/ontology#Url" or pro_info[rel_pq]["type"] == "http://wikiba.se/ontology#CommonsMedia": continue
                    rel_pq_name = pro_info[rel_pq]['label']
                    if pro_info[rel_pq]["type"] == "http://wikiba.se/ontology#Time" and not obj.startswith("_:"):
                        t_rels.add(rel_pq_name)
                        obj = obj.replace("T00:00:00Z", "")
                        date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_pq_name})
                        t_ens.add(obj)
                if "T00:00:00Z" in obj:
                    obj = obj.replace("T00:00:00Z","")
                    t_ens.add(obj)
                    date_in_f[statement_id]['date'].append({"date_id":obj, "date_rel":rel_pq_name})
                    t_rels.add(rel_pq_name)
                date_in_f[statement_id]['tuple'].append([
                    {"kb_id": statement_id, "text": statement_id},
                    {"rel_id": rel_pq, "text": rel_pq_name},
                    {"kb_id": obj, "text": obj_name}])

                ens.add(statement_id)
                ens.add(obj)
                rels.add(rel_pq_name)

    #print (t_rels)
    for statement_id in date_in_f:
        tuples.append((date_in_f[statement_id]['tuple'], date_in_f[statement_id]['date']))
    return tuples, t_rels, t_ens, rels, ens, tkgfacts

def _read_corners(cornerstone):
    """Return map from question ids to cornerstone entities."""
    corners = []
    for key in cornerstone:
        if key.strip().split("::")[1] == "Entity":
            corner = key.strip().split("::")[2]
            if "T00:00:00Z" in corner:
                corner = corner.replace("T00:00:00Z","")
            elif NUM_PATTERN.match(corner):
                corner = corner.lstrip("+")
            corners.append(corner)
    return corners

def _readable_entities(entities, weight):
    readable_entities = []
    weight_normalize = {}
    for key in weight:
        if "T00:00:00Z" in key:
            key_norm = key.replace("T00:00:00Z", "")
        elif NUM_PATTERN.match(key):
            key_norm = key.lstrip("+")
        else:
            key_norm = key
        weight_normalize[key_norm] = weight[key]

    for ent in entities:
        sc = 0.0
        if ent in weight_normalize:
            sc = weight_normalize[ent]
        readable_entities.append(
                {"text": ent, "kb_id": ent,
                    "score": sc})

    return readable_entities

def _get_answer_coverage(GT, entities):
    found, total = 0., 0
    #print ("\nentities:")
    #print (entities)
    print("\n\nanswer:")
    for answer in GT:
        #print (answer)
        if answer[0] in entities:
            found += 1.
            #print (GT)
        elif answer[0] + "T00:00:00Z" in [item.lower() for item in entities]:
            found += 1.
            #print(GT)

        total += 1
    return found / total

def get_qtkg(tempstate):
    tkgfacts = set()
    for statement_id in tempstate.keys():
        for item in tempstate[statement_id]['date']:
            if item[1].startswith('-'):
                int_date = int('-' + item[1].strip().replace('-', ''))
            else:
                int_date = int(item[1].strip().replace('-', ''))
            ps_spo = tempstate[statement_id]['ps_spo'][0]
            if (ps_spo[2], ps_spo[3]) in tempstate[statement_id]['date']:
                if ps_spo[3].startswith('-'):
                    int_date = int('-' + ps_spo[3].strip().replace('-', ''))
                else:
                    int_date = int(ps_spo[3].strip().replace('-', ''))
                tuple = (ps_spo[0], ps_spo[2], ps_spo[3], ps_spo[2], ps_spo[3], int_date)
                tkgfacts.add(tuple)
            else:
                tuple = (ps_spo[0], ps_spo[2], ps_spo[3], item[0], item[1], int_date)
                tkgfacts.add(tuple)
            for pq_spo in tempstate[statement_id]['pq_spo']:
                if (pq_spo[2], pq_spo[3]) not in tempstate[statement_id]['date']:
                    tuple = (pq_spo[0], pq_spo[2], pq_spo[3], item[0], item[1], int_date)
                    tkgfacts.add(tuple)
    tkgfacts = sorted(list(tkgfacts),key=lambda x:x[5])

    return tkgfacts

def _read_weight_from_QKG(graph_file):
    G = nx.read_gpickle(graph_file)
    return {G.nodes[n]['id']: G.nodes[n]['weight'] for n in G}

def _read_seeds(instance, nerd):
    """Return map from question ids to seed entities."""
    seeds = set()
    wiki_ids = list()
    wiki_ids += [item[0] for item in instance["elq"]]
    if nerd == "elq-wat":
        wiki_ids += [item[0][0] for item in instance["wat"]]
    elif nerd == "elq-tagme":
        wiki_ids += [item[0] for item in instance["tagme"]]
    elif nerd == "elq-tagme-wat":
        wiki_ids += [item[0][0] for item in instance["wat"]]
        wiki_ids += [item[0] for item in instance["tagme"]]

    return list(set(wiki_ids))

def get_extractentities_spo(spos):
    spo_entity = list()
    # if os.path.exists(spo_file):
    #     print(spo_file + ' exists.')
    # else:
    #     print(spo_file + ' not found!')
    #     return spo_entity
    # f22 = open(spo_file, 'r')
    # spos = f22.readlines()
    # f22.close()
    for line in spos:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        sub = triple[1]
        obj = triple[5]
        sub_name = triple[2]
        obj_name = triple[6]
        if 'corner#' in sub:
            sub = sub.replace('corner#', '').split('#')[0]
        if 'corner#' in obj:
            obj = obj.replace('corner#', '').split('#')[0]
        if {"id": sub, "label":sub_name} not in spo_entity:
            spo_entity.append({"id": sub, "label":sub_name})
        if {"id": obj, "label":obj_name} not in spo_entity:
            spo_entity.append({"id": obj, "label":obj_name})

    return spo_entity

def get_subgraph(dataset, input_path, output_path, pro_info, cfg):
    t1 = time.time()
    nerd = cfg["nerd"]

    seed_map = {}
    corner_map = {}
    answer_recall = []
    total = 0
    bad_questions = []
    good_questions = []
    ok_questions = []
    SPONOTFOUND = 0
    total_entities = 0

    with open(input_path, "r") as fi, open(output_path, "wb") as fo:
        for line in tqdm(fi):
            instance = json.loads(line)
            QuestionId = instance["Id"]
            QuestionText = instance["Question"]
            comgst_spo_list = instance["complete_gst_spo_list"]
            cornerstone = instance["cornerstone"]
            entities_weight = instance["entity_weight"]
            enhance_facts = []
            for evidence in instance["temporal_evidences"]:
                enhance_facts += evidence["statement_spo_list"]

            hit_enhance_spos = answer_presence_gst(get_extractentities_spo(enhance_facts), instance["answers"])
            hit_spos = answer_presence_gst(get_extractentities_spo(comgst_spo_list), instance["answers"])
            answer_presence = hit_enhance_spos or hit_spos
            GT = get_answer(instance["answers"])
            seed_map[QuestionId] = _read_seeds(instance, nerd)
            corner_map[QuestionId] = _read_corners(cornerstone)
            tuples, t_rels, t_ents, rels, ents, tkgfacts = get_triple_from_SPO(comgst_spo_list, enhance_facts, pro_info)

            seed_entities = []
            corner_entities = []
            ans_entities = []
            for ee in corner_map[QuestionId]:
                if ee in ents:
                    corner_entities.append(ee)
            for ee in seed_map[QuestionId]:
                if ee in ents:
                    seed_entities.append(ee)

            if corner_entities:
                for answer in GT:
                    if answer[0] in ents:
                        ans_entities.append(answer[0])

                # answer_candidate_ids = {answer_candidate_id: answer_candidate_id.lower().strip().replace('"', "").replace("+", "") for
                #     answer_candidate_id in ents}
                #
                # for answer in GT:
                #     gold_answer_id = answer[0].lower().strip().replace('"', "").replace("+", "")
                #     for key, value in answer_candidate_ids.items():
                #         if value == gold_answer_id:
                #             ans_entities.append(key)

            if not instance["answers"] or len(ans_entities) == 0:
                curr_recall = 0
            elif len(ans_entities) > 0:
                curr_recall = 1

            if curr_recall == 0:
                bad_questions.append(QuestionId)
            elif curr_recall == 1:
                ok_questions.append(QuestionId)

            if answer_presence and curr_recall == 0:
                print("\nans_entities")
                print(ans_entities)
                print("\ncorner_entities")
                print(corner_entities)
                print("\nGT")
                print(GT)
                print("\nanswer_presence")
                print(answer_presence)
                facts_ent = []
                facts_ans = []
                for item in get_extractentities_spo(enhance_facts):
                    facts_ent.append(item["id"])
                for item in get_extractentities_spo(comgst_spo_list):
                    facts_ent.append(item["id"])
                for item in GT:
                    if item[0] in facts_ent:
                        facts_ans.append(item)
                print(ents)
                print(facts_ent)

            answer_recall += [curr_recall]

            total += 1

            answers = [{"kb_id": answer[0], "text": answer[1]} for answer in GT]

            ques_entities = _readable_entities(ents, entities_weight)
            total_entities += len(ques_entities)
            data = {
                "question": QuestionText,
                "question_seed_entities": seed_map[QuestionId],
                "seed_entities": _readable_entities(seed_entities, entities_weight),
                "corner_entities": _readable_entities(corner_entities, entities_weight),
                "answers": answers,
                "id": QuestionId,
                "subgraph": {
                    "entities": ques_entities,
                    "tuples": tuples
                },
                "signal": instance["Temporal signal"],
                "type": instance["Temporal question type"],
                "tkg": tkgfacts,
                "tempentities": list(t_ents),
                "temprelations": list(t_rels)
                }

            if dataset == 'train':
                if data['id'] in ok_questions:
                    fo.write(json.dumps(data).encode("utf-8"))
                    fo.write("\n".encode("utf-8"))
            else:
                fo.write(json.dumps(data).encode("utf-8"))
                fo.write("\n".encode("utf-8"))

            result = "{0}|{1}|{2}|{3}|{4}|{5}".format(
                str(QuestionId),
                QuestionText,
                str(len(ents)),
                str(len(corner_entities)),
                str(len(seed_entities)),
                str(curr_recall)
                )
            print (result)

    t2 = time.time()
    print("total number of questions: " + str(total))
    print("total number of entities: " + str(total_entities))
    print("average number of entities: " + str(total_entities * 1.0 / (total - SPONOTFOUND)))
    print("questions with empty subgraphs: " + str(SPONOTFOUND) )
    print("Good questions =  " + str(len(ok_questions) * 1.0 / total) )
    print("Number of ok questions =  " + str(len(ok_questions)))
    print("Answer recall =  " + str(sum(answer_recall) / len(answer_recall)))
    print("total time: " + str(t2-t1))
    print("average time: " + str((t2-t1) * 1.0 / total) )


# if __name__ == "__main__":
#     import argparse
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('-d', '--dataset', type=str, default='dev')
#     argparser.add_argument('-f', '--topf', type=int, default=25)
#     argparser.add_argument('-g', '--topg', type=int, default=25)
#     argparser.add_argument('-t', '--topt', type=int, default=25)
#     cfg = globals.get_config(globals.config_file)
#     pro_info = globals.ReadProperty.init_from_config().property
#     args = argparser.parse_args()
#     dataset = args.dataset
#     topf = args.topf
#     topg = args.topg
#     topt = args.topt
#
#     get_subgraph(dataset, pro_info, cfg, topf, topg, topt)


