"""Script to get temporal facts for entities in top-g GSTs.
"""
import os
import json
import globals
import time
import re
from tqdm import tqdm
from get_CLOCQ_Wikidata_SPOs import get_wikidata_temptuplesfromclocq

def get_entities_spo(spos):
    spo_entity = list()
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

def write_tempfact_to_lines(tempfact, ids):
    templines = []
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
        ps_line = sta + "-ps:" + "||" + p + "||" + pre + "||" + obj + "||" + str(objname) + '\n'
        templines.append(ps_line)
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
            pq_line = sta + "-pq:" + "||" + p + "||" + pre + "||" + obj + "||" + objname + '\n'
            if pq_line not in templines:
                templines.append(pq_line)
    return templines

if __name__ == "__main__":
    import argparse
    cfg = globals.get_config(globals.config_file)
    pro_info = globals.ReadProperty.init_from_config().property
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='test')
    argparser.add_argument('-o', '--option', type=str, default='xg')
    argparser.add_argument('-f', '--topf', type=int, default=25)
    argparser.add_argument('-g', '--topg', type=int, default=10)
    argparser.add_argument('-c', '--connect', type=str, default='first')
    args = argparser.parse_args()
    dataset = args.dataset
    topf = args.topf
    topg = args.topg
    option = args.option
    connect = args.connect
    if option == "qkg":
        topg = 0
    test = cfg["data_path"] + cfg["test_data"]
    dev = cfg["data_path"] + cfg["dev_data"]
    train = cfg["data_path"] + cfg["train_data"]
    if dataset == 'test': in_file = test
    elif dataset == 'dev': in_file = dev
    elif dataset == 'train': in_file = train

    t1 = time.time()
    datas = []
    spo_file = cfg["data_path"] + connect + "_" + option + "/" + dataset + '_' + str(topf) + '_' + str(topg) + ".json"
    tempspo_file = cfg["data_path"] + connect + "_" + option + "/" + dataset + '_' + str(topf) + '_' + str(topg) + '_temp.json'

    #spo_file = cfg["data_path"] + option + "/" + dataset + '_' + str(topf) + '_' + str(topg) + ".json"
    #tempspo_file = cfg["data_path"] + option + "/" + dataset + '_' + str(topf) + '_' + str(topg) + '_temp.json'

    f1 = open(tempspo_file, "wb")
    with open(spo_file, encoding='utf-8') as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)
            datas.append(line)

    for question in datas:
        QuestionId = question["id"]
        QuestionText = question["question"]
        print("\n\nQuestion Id-> ", QuestionId)
        print("Question -> ", QuestionText)
        spo = question["subgraph"]
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        unionGST_entities = get_entities_spo(spo)
        qids = [item for item in unionGST_entities if ENT_PATTERN.match(item) != None]
        print (len(qids))
        tempfact = get_wikidata_temptuplesfromclocq(qids, pro_info)
        templines = write_tempfact_to_lines(tempfact, qids)
        temp = {
        "question": QuestionText,
        "id": QuestionId,
        "tempfact": templines
        }
        f1.write(json.dumps(temp).encode("utf-8"))
        f1.write("\n".encode("utf-8"))
    t2 = time.time()
    total_t = t2 - t1
    print("\n\ntotal_t: " + str(total_t))
    print("\n\naverage time: " + str(total_t / len(datas)))
    f1.close()

