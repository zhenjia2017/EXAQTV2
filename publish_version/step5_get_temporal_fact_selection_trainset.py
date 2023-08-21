import yaml
import os
import json
import random
import csv
import globals
import truecase
import pickle

from get_CLOCQ_Wikidata_SPOs import write_clocqtempspo_to_file, get_wikidata_twohoptemptuplesfromclocq, get_wikidata_temptuplesfromclocq

cfg = globals.get_config(globals.config_file)
dev = cfg["data_path"] + cfg["dev_data"]
train = cfg["data_path"] + cfg["train_data"]
DATASETs = [dev, train]
pro_info = globals.ReadProperty.init_from_config().property
TRAINING_FILE = cfg["data_path"] + "phase2_temporal_fact_selection_trainset.csv"
train_list = []
ques_answer_1hop = cfg["data_path"] + "ques_answer_1hoptemp.pkl"
ques_answer_2hop = cfg["data_path"] + "ques_answer_2hoptemp.pkl"

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

def get_extractentities_spo(spo_file):
    spo_entity = list()
    spo_entity_upper = list()
    # print(spo_file)
    if not os.path.exists(spo_file):
        print(spo_file + ' not found!')
        return spo_entity
    f22 = open(spo_file, 'r')
    spos = f22.readlines()
    f22.close()
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
        spo_entity_upper.append(sub)
        spo_entity_upper.append(obj)
        spo_entity.append(sub.lower())
        spo_entity.append(obj.lower())
    return list(set(spo_entity)), list(set(spo_entity_upper))

def get_twohoptempspo_from_clocq(spo_temp_file, wiki_ids, pro_info):
    seeds = set()
    for item in wiki_ids:
        seeds.add(item[0])
    tempfact = get_wikidata_twohoptemptuplesfromclocq(list(seeds), pro_info)
    write_clocqtempspo_to_file(tempfact, spo_temp_file, list(seeds))
    print("\n\n2-hop temporal SPOs generated...")
    return seeds

def get_tempspo_from_clocq(tagme_wiki_ids_file, elq_wiki_ids_file, spo_temp_file, pro_info):
    wiki_ids = set()
    seeds = set()
    if os.path.exists(tagme_wiki_ids_file):  # line A
        #print('TagMe seed entity' + ' exists.')
        with open(tagme_wiki_ids_file) as f:
            for line in f:
                entity, score, text = line.strip().split('\t')
                wiki_ids.add((entity, score, text))
                seeds.add(entity)
    if os.path.exists(elq_wiki_ids_file):  # line A
        #print('ELQ seed entity' + ' exists.')
        with open(elq_wiki_ids_file) as f:
            for line in f:
                entity, score, text = line.strip().split('\t')
                wiki_ids.add((entity, score, text))
                seeds.add(entity)

    tempfact = get_wikidata_temptuplesfromclocq(list(seeds), pro_info)
    write_clocqtempspo_to_file(tempfact, spo_temp_file, list(seeds))
    print("\n\ntemporal SPOs generated...")
    return seeds, wiki_ids

count = 0
answer_in_1hop = []
answer_in_2hop = []
answer_in_nontemp = []
for file in DATASETs:
    data = json.load(open(file))
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

        id_path = cfg['path'] + '/files' + '/ques_' + ques_id
        tagme_wiki_ids_file = id_path + '/wiki_ids_tagme_new.txt'
        elq_wiki_ids_file = id_path + '/wiki_ids_elq.txt'
        spo_file = id_path + '/' + 'SPO_new.txt'
        tempSPO_file = id_path + '/' + 'tempSPO_tagme_elq_new.txt'
        seeds, wiki_ids = get_tempspo_from_clocq(tagme_wiki_ids_file, elq_wiki_ids_file, tempSPO_file, pro_info)
        have_answer_file = None
        spo_sta = {}
        ans_sta = {}
        main_rel_sta = {}
        if not (os.path.exists(tempSPO_file)):
            print ("tempSPO_file not exist!")
            continue
        temp_entities, temp_entities_upper = get_extractentities_spo(tempSPO_file)
        tempspo_ac = _get_answer_coverage(GT, temp_entities)
        if tempspo_ac > 0:
            have_answer_file = tempSPO_file
            print("\nanswer in one hop temporal facts! " + str(ques_id))
            answer_in_1hop.append(ques_id)
        elif tempspo_ac <= 0:
            file_entities, file_entities_upper = get_extractentities_spo(spo_file)
            spo_ac = _get_answer_coverage(GT, file_entities)
            if spo_ac > 0:
            #    have_answer_file = spo_file
                answer_in_nontemp.append(ques_id)
                print("\nanswer in one hop non-temporal facts! " + str(ques_id))
            else:
                twohoptempspo_file = id_path + '/tempSPO_tagme_elq_2hop.txt'
                get_twohoptempspo_from_clocq(twohoptempspo_file, wiki_ids, pro_info)
                temp_entities_2hop, temp_entities_upper_2hop = get_extractentities_spo(twohoptempspo_file)
                tempspo_ac_2hop = _get_answer_coverage(GT, temp_entities_2hop)
                print("\nanswer recall in two hop facts! " + str(tempspo_ac_2hop))
                if tempspo_ac_2hop > 0:
                    have_answer_file = twohoptempspo_file
                    answer_in_2hop.append(ques_id)
                    print("\nanswer in two hop temporal facts! "  + str(ques_id))

        if not have_answer_file: continue

        with open(have_answer_file) as f:
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
                if len(GT) > 0 and ent in GT:
                    pos_flag = 1
                    break
            if pos_flag == 1:
                positive.append(statement_id)
                pos_rel.append(main_rel_sta[statement_id])

        for statement_id in ans_sta:
            pos_flag = 0
            for ent in ans_sta[statement_id]:
                if ent in GT:
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
        writer.writerow(sample)

pickle.dump(answer_in_1hop, open(ques_answer_1hop, 'wb'))
pickle.dump(answer_in_2hop, open(ques_answer_2hop, 'wb'))
print("total of questions having positive samples", str(count))
print("total of questions having positive samples in 1-hop temporal facts", str(len(answer_in_1hop)))
print("total of questions having positive samples in 2-hop temporal facts", str(len(answer_in_2hop)))
print("total of questions having positive samples in non-temporal facts", str(len(answer_in_nontemp)))

