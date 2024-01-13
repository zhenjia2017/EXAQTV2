import sys
import json
from tqdm import tqdm
from util import get_config

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
            hits5 = 1.0  # hits@5
    else:
        for j in range(0, len(sorted_l)):
            if j < topk:
                best5_pred.append(sorted_l[j][0] + '#' + str(sorted_l[j][1]))
                if sorted_l[j][0] in answers:
                    hits5 = 1.0

    return hits5, best5_pred

# presicion@1 recall@1 f1@1
def get_prfmetric(kb_pred_file, topk):
    pred_list = []
    with open(kb_pred_file) as f_kb:
        line_id = 0
        for line_kb in tqdm(zip(f_kb)):
            line_id += 1
            #print (line_kb[0])
            line_kb = json.loads(line_kb[0])
            #print (line_kb['answers'])
            answers = set([answer.lower() for answer in line_kb['answers']])
            # total_not_answerable += (len(answers) == 0)
            # assert len(answers) > 0

            dist_kb = line_kb['dist']
            kb_entities = set(dist_kb.keys())
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    rank = 0
    pred_list1 = []
    for j in range(0, len(pred_list)):
        if j > 0:
            if pred_list[j][2] < pred_list[j - 1][2]:
                rank += 1
        # answer_list[j]=tuple(list(answer_list[j]).append(rank))
        pred_list1.append((pred_list[j][0], rank))
    answers_lower = []
    for item in answers:
        answers_lower.append(item.lower())
    correct, total = 0.0, 0.0
    pred_list = pred_list1
    for i in range(0, len(pred_list)):
        if pred_list[i][1] <= 1:  # rank 1
            ans1 = pred_list[i][0]
    #ans1 = answer_list[0][0].split('|')
            if 'T00:00:00Z' in ans1: ans1 = ans1.replace('T00:00:00Z', '')
            if ans1.lower() in answers_lower:
                correct += 1
            total += 1
    if len(answers_lower) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0  # precision, recall, f1
        else:
            return 0.0, 1.0, 0.0  # precision, recall, f1
    else:
        if total == 0:
            return 1.0, 0.0, 0.0  # precision, recall, f1
        else:
            precision, recall = correct / total, correct / len(answers_lower)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1

def get_mmr_metric(kb_entities, dist_kb, topk, answers):
    pred_list = []
    for entity in kb_entities:
        pred_list.append((entity, dist_kb[entity]))
    sorted_l = sorted(pred_list, reverse=True, key=lambda t: t[1])
    # rank = 0
    # pred_list1 = []
    # for j in range(0, len(pred_list)):
    #     if j > 0:
    #         if pred_list[j][1] < pred_list[j - 1][1]:
    #             rank += 1
    #     pred_list1.append((pred_list[j][0], rank))

    #pred_list = pred_list1

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

def get_one_f1(entities, dist, eps, answers):
    #correct = 0.0
    #total = 0.0
    best_entity = -1
    max_prob = 0.0
    preds = []
    for entity in entities:
        if dist[entity] > max_prob:
            max_prob = dist[entity]
            best_entity = entity
        if dist[entity] > eps:
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
            return 0.0, 1.0, 0.0, 1.0  # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits

def compare_pr(kb_pred_file, eps_kb, fp):
    #doc_only_recall, doc_only_precision, doc_only_f1, doc_only_hits = [], [], [], []
    kb_only_recall, kb_only_precision, kb_only_f1, kb_only_hits = [], [], [], []
    kb_only_mrr = []
    kb_only_hits5 = []
    kb_only_p1  = []

    #ensemble_recall, ensemble_precision, ensemble_f1, ensemble_hits = [], [], [], []
    #hybrid_recall, hybrid_precision, hybrid_f1, hybrid_hits = [], [], [], []
    #ensemble_all_recall, ensemble_all_precision, ensemble_all_f1, ensemble_all_hits = [], [], [], []

    total_not_answerable = 0.0
    with open(kb_pred_file) as f_kb:
        line_id = 0
        for line_kb in tqdm(zip(f_kb)):
            line_id += 1
            #print (line_kb[0])
            line_kb = json.loads(line_kb[0])
            #print (line_kb['answers'])
            answers = set([answer for answer in line_kb['answers']])
            # total_not_answerable += (len(answers) == 0)
            # assert len(answers) > 0
            id = line_kb['id']
            dist_kb = line_kb['dist']
            kb_entities = set(dist_kb.keys())

            p, r, f1, hits1, best_entity = get_one_f1(kb_entities, dist_kb, eps_kb, answers)
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
            #print(result)
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



if __name__ == "__main__":

    path = '/GW/qa2/work/exact/GRAFTNET'
    test_subg = 'test_v4c_TAGMEELQSubgraphSPORelWM500.json'
    dev_subg = 'dev_v4c_TAGMEELQSubgraphSPORelWM500.json'
    train_subg = 'train_v4c_TAGMEELQSubgraphSPORelWM500.json'
    ADD_ENT_MAX = '500'
    epochs = [1, 5, 10, 15, 20, 25, 50, 100]
    minibatch = [10, 25, 50, 100, 200]

    for epoch in epochs:
        for batch in minibatch:
            config_file = path + '/config/' + 'configAnswer_v4c_TAGMEELQM500_baseline_' + str(epoch) + '_' + str(
                batch) + '.yml'
            CFG = get_config(config_file)
            pred_kb_file = CFG['model_folder'] + CFG['pred_file']
            result_path = '/GW/qa2/work/exact/GRAFTNET/result/'
            re_fp = result_path + config_file.split('/')[len(config_file.split('/')) - 1].replace('.yml', 'score.txt')

            fp = open(re_fp, 'w', encoding='utf-8')
            eps_kb = 0.2
            compare_pr(pred_kb_file, eps_kb, fp)

            fp.close()

    # #original question with type and signal, temporal relation attention
    # #config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_gqrt_ts.yml'
    # #config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_oqrt_fact.yml'
    # config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_oqrM100.yml'
    #
    # # original question, temporal relation attention
    # #config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_oqrt.yml'
    # # original question, relation attention
    # #config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_oqr.yml'
    # # original question with type and signal, relation attention
    # #config_file = '/GW/qa2/work/exact/GRAFTNET/config/configAnswer_v2_oqr_ts.yml'
    # result_path = '/GW/qa2/work/exact/QUEST_EXACT/result/'
    # re_fp = result_path  + config_file.split('/')[len(config_file.split('/'))-1].replace('.yml','.txt')
    # CFG = get_config(config_file)
    # #best_pred_file =
    # #dataset = sys.argv[1]
    # #pred_kb_file = sys.argv[2]
    # pred_kb_file = CFG['model_folder']+CFG['pred_file']
    # #re_fp = CFG['model_folder'] + CFG['metric_file']
    # fp = open(re_fp, 'w', encoding='utf-8')
    # eps_kb = 0.2
    # # if dataset == "tempq":
    # #     eps_kb = 0.2
    # # else:
    # #     assert False, "dataset not recognized"
    # #pred_kb_file = "/GW/qa/work/exact/GRAFTNET/model/pred_kb_spo_tem_haveanswer"
    #
    # compare_pr(pred_kb_file, eps_kb, fp)
    # fp.close()
