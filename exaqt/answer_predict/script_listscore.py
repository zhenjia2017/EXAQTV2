import json
from tqdm import tqdm

def combine_dist(dist1, dist2, w1):
    ensemble_dist = dist2.copy()
    for gid, prob in dist1.items():
        if gid in ensemble_dist:
            ensemble_dist[gid] = (1 - w1) * ensemble_dist[gid] + w1 * prob
        else:
            ensemble_dist[gid] = prob
    return ensemble_dist

#hits@5
def get_hitsmetric(kb_entities, dist_kb, topk, answers):
    pred_list = []
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    sorted_l = sorted(pred_list, reverse=True, key=lambda t: t[1])
    hits5 = 0.0
    best5_pred = []
    if len(answers) == 0:
        if len(kb_entities) == 0:
            hits5 = 1.0  # hits@5
        else:
            hits5 = 0.0  # hits@5
    else:
        for j in range(0, len(sorted_l)):
            if j < topk:
                best5_pred.append(sorted_l[j][0] + '#' + str(sorted_l[j][1]))
                if sorted_l[j][0] in answers:
                    hits5 = 1.0

    return hits5, best5_pred

def get_mmr_metric(kb_entities, dist_kb, topk, answers):
    pred_list = []
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    sorted_l = sorted(pred_list, reverse=True, key=lambda t: t[1])

    mrr = 0.0
    flag = 0
    if topk == 5:
        topk = len(pred_list)

    for i in range(0, len(sorted_l)):
        for answer in answers:
            if answer.lower() == sorted_l[i][0].lower():  # Order preserving
                mrr += 1.0 / float(i+1)
                flag = 1
                break
        if flag == 1:
            break

    return mrr

def get_one_f1(entities, dist, threshold, answers):
    best_entity = -1
    max_prob = 0.0
    preds = []
    for entity in entities:
        if dist[entity] > max_prob:
            max_prob = dist[entity]
            best_entity = entity
        if dist[entity] > threshold:
            preds.append(entity)
    precision, recall, f1, hits = cal_eval_metric(best_entity, preds, answers)
    return  precision, recall, f1, hits, best_entity

def cal_eval_metric(best_pred, preds, answers):
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0  # precision, recall, f1, hits
        else:
            return 0.0, 0.0, 0.0, 0.0  # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits

def evaluate_result_for_category(nerd_file, test_re_fp, out_file):
    fo = open(out_file, 'w', encoding='utf-8')
    result = open(test_re_fp, 'r', encoding='utf-8')
    res_lines = result.readlines()
    result_number = {}

    with open(nerd_file, "r") as fp:
        data = json.load(fp)

    type_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal':[]}
    p1_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal':[]}
    h5_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal':[]}
    mrr_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal':[]}

    count = 0
    for instance in tqdm(data):
        if type(instance["Temporal question type"]) != list:
            instance["Temporal question type"] = [instance["Temporal question type"]]
        types = instance["Temporal question type"]
        #if "Ordinal" in types: continue
        id = str(instance["Id"])
        for item in types:
            type_dic[item].append(id)
        count += 1

    for line in res_lines:
        if '|' in line and len(line.split('|')) > 3:
            # print (line)
            id = line.split('|')[0]
            p1 = float(line.split('|')[1])
            h5 = float(line.split('|')[2])
            mrr = float(line.split('|')[3])

            for key, value in type_dic.items():
                if id in value:
                    p1_res_dic[key].append(p1)
                    h5_res_dic[key].append(h5)
                    mrr_res_dic[key].append(mrr)

    for key in p1_res_dic.keys():
        if key not in result_number:
            result_number[key] = {}
        result_number[key]['p1'] = str(round(sum(p1_res_dic[key]) / len(p1_res_dic[key]), 3))
        print('Average  hits1: ', str(sum(p1_res_dic[key]) / len(p1_res_dic[key])))

    for key in mrr_res_dic.keys():
        print(key)
        result_number[key]['mrr'] = str(round(sum(mrr_res_dic[key]) / len(mrr_res_dic[key]), 3))
        print('Average  mrr: ', str(sum(mrr_res_dic[key]) / len(mrr_res_dic[key])))

    for key in h5_res_dic.keys():
        print(key)
        result_number[key]['hits5'] = str(round(sum(h5_res_dic[key]) / len(h5_res_dic[key]), 3))

        print('Average  hits5: ', str(sum(h5_res_dic[key]) / len(h5_res_dic[key])))

    for key in result_number:
        fo.write(key + '|')
        for item in result_number[key]:
            fo.write(result_number[key][item])
            fo.write('|')
    fo.write('\n')
    fo.close()

def compare_pr(kb_pred_file, threshold, fp):
    kb_only_recall, kb_only_precision, kb_only_f1, kb_only_hits = [], [], [], []
    kb_only_mrr = []
    kb_only_hits5 = []

    with open(kb_pred_file) as f_kb:
        line_id = 0
        for line_kb in tqdm(zip(f_kb)):
            line_id += 1

            line_kb = json.loads(line_kb[0])

            answers = set([answer for answer in line_kb['answers']])

            id = line_kb['id']
            dist_kb = line_kb['dist']
            kb_entities = set(dist_kb.keys())

            p, r, f1, hits1, best_entity = get_one_f1(kb_entities, dist_kb, threshold, answers)
            kb_only_precision.append(p)
            kb_only_recall.append(r)
            kb_only_f1.append(f1)
            kb_only_hits.append(hits1)

            mmr = get_mmr_metric(kb_entities, dist_kb, 5, answers)
            kb_only_mrr.append(mmr)

            hits5, best_entities = get_hitsmetric(kb_entities, dist_kb, 5, answers)
            kb_only_hits5.append(hits5)

            result = "{0}|{1}|{2}|{3}|{4}".format(
                str(id), str(hits1), str(hits5), str(mmr), ";".join(best_entities))

            fp.write(result)
            fp.write("\n")
    fp.write ("line count |" + str(line_id))
    fp.write('\n')
    fp.write('Average hits1 |' + str(sum(kb_only_hits) / len(kb_only_hits)))
    fp.write('\n')
    fp.write('Average hits5 |' + str(sum(kb_only_hits5) / len(kb_only_hits5)))
    fp.write('\n')
    fp.write('Average mmr |'+ str(sum(kb_only_mrr) / len(kb_only_mrr)))
    fp.write('\n')
    fp.write('precision |' + str(sum(kb_only_precision) / len(kb_only_precision)))
    fp.write('\n')
    fp.write('recall |' + str(sum(kb_only_recall) / len(kb_only_recall)))
    fp.write('\n')
    fp.write('f1 |' + str(sum(kb_only_f1) / len(kb_only_f1)))
    fp.write('\n')

    print('Average hits1: ' , str(sum(kb_only_hits) / len(kb_only_hits)))
    print('Average hits5: ' , str(sum(kb_only_hits5) / len(kb_only_hits5)))
    print('Average mmr: ' , str(sum(kb_only_mrr) / len(kb_only_mrr)))
    print('precision: ' , str(sum(kb_only_precision) / len(kb_only_precision)))
    print('recall: ' , str(sum(kb_only_recall) / len(kb_only_recall)))
    print('f1: ' , str(sum(kb_only_f1) / len(kb_only_f1)))
