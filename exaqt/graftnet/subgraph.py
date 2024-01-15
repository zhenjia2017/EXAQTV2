"""Script to run PPR on question subgraphs and retain top entities.
"""


import random
import numpy as np
import pickle as pkl
import json
import os
import re
from tqdm import tqdm

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from exaqt.library.utils import get_logger

random.seed(0)

MAX_FACTS = 5000000
MAX_ITER = 20
RESTART = 0.8
MAX_ENT = 500
NOTFOUNDSCORE = 0.
EXPONENT = 2.
MAX_SEEDS = 20
DECOMPOSE_PPV = True
SEED_WEIGHTING = True
SEED_WEIGHTING_MODE = 'PPR'
RELATION_WEIGHTING = True
FOLLOW_NONCVT = True
USEANSWER = False
SPONOTFOUND = 0
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

def _personalized_pagerank(seed, W):
    """Return the PPR vector for the given seed and adjacency matrix.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.

    Returns:
        ppr: A vector of size E.
    """
    restart_prob = RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for i in range(MAX_ITER):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: break
        r = r_new
    return np.squeeze(s_ovr)

def _get_subgraph(entities, kb_r, multigraph_W):
    """Get subgraph describing a neighbourhood around given entities."""
    seed = np.zeros((multigraph_W.shape[0], 1))
    if not SEED_WEIGHTING:
        seed[entities] = 1. / len(set(entities))
    else:
        seed[entities] = np.expand_dims(np.arange(len(entities), 0, -1),
                                        axis=1)
        seed = seed / seed.sum()
    ppr = _personalized_pagerank(seed, multigraph_W)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted_ents = sorted_idx[:MAX_ENT]
    extracted_scores = ppr[sorted_idx[:MAX_ENT]]
    # check if any ppr values are nearly zero
    zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted_ents = extracted_ents[:zero_idx[0]]
    extracted_tuples = []
    ents_in_tups = set()
    for relation in kb_r:
        submat = kb_r[relation][extracted_ents, :]
        submat = submat[:, extracted_ents]
        row_idx, col_idx = submat.nonzero()
        for ii in range(row_idx.shape[0]):
            extracted_tuples.append(
                (extracted_ents[row_idx[ii]], relation,
                 extracted_ents[col_idx[ii]]))
            ents_in_tups.add((extracted_ents[row_idx[ii]],
                extracted_scores[row_idx[ii]]))
            ents_in_tups.add((extracted_ents[col_idx[ii]],
                extracted_scores[col_idx[ii]]))
    return extracted_tuples, list(ents_in_tups)

def _read_facts(data, relation_embeddings, question_embedding):

    """Read all triples from the fact file and create a sparse adjacency
    matrix between the entities. Returns mapping of entities to their
    indices, a mapping of relations to the
    and the combined adjacency matrix."""
    entity_map = {}
    relation_map = {}
    row_ones, col_ones = [], []
    row_weight_ones, col_weight_ones = [], []
    num_entities = 0
    num_facts = 0
    weight = {}
    #print (qId)
    for statement in data:
        for (e1, rel, e2) in data[statement]:
            #print (sub, rel, obj)

            if e1 not in entity_map:
                entity_map[e1] = num_entities
                num_entities += 1
            if e2 not in entity_map:
                entity_map[e2] = num_entities
                num_entities += 1
            if rel not in relation_map:
                relation_map[rel] = [[], []]
            row_ones.append(entity_map[e1])
            col_ones.append(entity_map[e2])
            row_ones.append(entity_map[e2])
            col_ones.append(entity_map[e1])
            relation_map[rel][0].append(entity_map[e1])
            relation_map[rel][1].append(entity_map[e2])
            num_facts += 1
            if num_facts == MAX_FACTS:
                break
    if not relation_map:
        return {}, {}, None
    for rel in relation_map:
        row_ones, col_ones = relation_map[rel]
        m = csr_matrix(
            (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
            shape=(num_entities, num_entities))
        relation_map[rel] = normalize(m, norm="l1", axis=1)
        if RELATION_WEIGHTING:
            if rel not in relation_embeddings:
                score = NOTFOUNDSCORE
            else:
                score = np.dot(question_embedding, relation_embeddings[rel]) / (np.linalg.norm(question_embedding) * np.linalg.norm(relation_embeddings[rel]))
            relation_map[rel] = relation_map[rel] * np.power(score, EXPONENT)

    # if DECOMPOSE_PPV:
    #     adj_mat = sum(relation_map.values()) / len(relation_map)
    # else:
    #     adj_mat = csr_matrix(
    #         (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
    #         shape=(num_entities, num_entities))

    if DECOMPOSE_PPV:
        adj_mat = sum(relation_map.values()) / len(relation_map)
    else:
        adj_mat = csr_matrix(
            (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
            shape=(num_entities, num_entities))
    print("\nlen(relation_map)")
    print(len(relation_map))
    adj_mat.data = np.nan_to_num(adj_mat.data)
    #save_npz('/GW/qa/work/exact/test/EXACT/data/sparse_matrix.npz', adj_mat)
    #normalize_adj_mat = normalize(adj_mat, norm="l1", axis=1)
    return entity_map, relation_map, normalize(adj_mat, norm="l1", axis=1)

def get_triple_from_SPO(evidences, pro_info):
    #get spo triples like graftnet
    #s:subject
    #o:intermediate if there is pq statement

    qual = dict()
    state = dict()
    entity_name = dict()
    t_rels = set()
    t_ens = set()
    line_spo_map = {}

    for evidence in evidences:
        for line in evidence["statement_spo_list"]:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = line.split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
            if statement_id not in state:
                state[statement_id] = dict()
                state[statement_id]['ps'] = []
                state[statement_id]['pq'] = []
            if "-ps:" in line.split("||")[0]:
                state[statement_id]['ps'].append(line)
            if "-pq:" in line.split("||")[0]:
                state[statement_id]['pq'].append(line)
            line_spo_map[line] = []

    for statement_id in state:
        if statement_id not in qual:
            qual[statement_id] = []
        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) == 0:
            ps_lines = state[statement_id]['ps']
            for line in ps_lines:
                sub = line.split("||")[1]
                sub_name = line.split("||")[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                if NUM_PATTERN.match(sub):
                    sub = sub.lstrip("+")
                #if line.split("|")[3] not in self.pro_info or line.split("|")[4] not in self.pro_info: continue
                rel = line.split("||")[3]
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time":
                        t_rels.add(rel_name)
                obj = line.split("||")[5]
                obj_name = line.split("||")[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                if "T00:00:00Z" in obj:
                    obj = obj.replace("T00:00:00Z","")
                    t_ens.add(obj)
                    t_rels.add(rel_name)
                if rel_name in t_rels and obj not in t_ens: t_ens.add(obj)
                # if rel_name in t_rels:
                #     print((sub, rel_name, obj))
                qual[statement_id].append((sub, rel_name, obj))
                line_spo_map[line].append((sub, rel_name, obj))
                entity_name[sub] = sub_name
                entity_name[obj] = obj_name
        if len(state[statement_id]['ps']) > 0 and len(state[statement_id]['pq']) > 0:
            ps_lines = state[statement_id]['ps']
            pq_lines = state[statement_id]['pq']
            for line in ps_lines:
                sub = line.split("||")[1]
                sub_name = line.split("||")[2]
                if "corner#" in sub: sub = sub.split("#")[1]
                if NUM_PATTERN.match(sub):
                    sub = sub.lstrip("+")
                # if line.split("||")[3] not in self.pro_info or line.split("||")[4] not in self.pro_info: continue
                rel = line.split("||")[3]
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time":
                        t_rels.add(rel_name)
                ps_obj = statement_id
                obj = line.split("||")[5]
                obj_name = line.split("||")[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                if "T00:00:00Z" in obj:
                    obj = obj.replace("T00:00:00Z","")
                    t_ens.add(obj)
                    t_rels.add(rel_name)
                if rel_name in t_rels and obj not in t_ens: t_ens.add(obj)

                qual[statement_id].append((sub, rel_name, ps_obj))
                qual[statement_id].append((ps_obj, rel_name, obj))
                line_spo_map[line].append((sub, rel_name, ps_obj))
                line_spo_map[line].append((ps_obj, rel_name, obj))
                entity_name[sub] = sub_name
                entity_name[obj] = obj_name
            for line in pq_lines:
                pq_sub = statement_id
                rel = line.split("||")[4]
                if rel in pro_info:
                    rel_name = pro_info[rel]['label']
                    if pro_info[rel]["type"] == "http://wikiba.se/ontology#Time":
                        t_rels.add(rel_name)
                obj = line.split("||")[5]
                obj_name = line.split("||")[6].replace("T00:00:00Z", "")
                if "corner#" in obj: obj = obj.split("#")[1]
                if NUM_PATTERN.match(obj):
                    obj = obj.lstrip("+")
                if "T00:00:00Z" in obj:
                    obj = obj.replace("T00:00:00Z","")
                    t_ens.add(obj)
                    t_rels.add(rel_name)
                if rel_name in t_rels and obj not in t_ens: t_ens.add(obj)

                qual[statement_id].append((pq_sub, rel_name, obj))
                line_spo_map[line].append((pq_sub, rel_name, obj))
                entity_name[obj] = obj_name
    return qual, t_rels, t_ens, line_spo_map, entity_name

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

def _read_corners(cornerstone_file):
    """Return map from question ids to cornerstone entities."""
    corners = []
    if not os.path.exists(cornerstone_file):
        print("cornerstone file not found!")
    else:
        data = pkl.load(open(cornerstone_file, 'rb'))

        for key in data:
            if key.strip().split("::")[1] == "Entity":
                corners.append(key.strip().split("::")[2])
    return corners

def _convert_to_readable(tuples, inv_map):
    readable_tuples = []
    for tup in tuples:
        readable_tuples.append([
            {"kb_id": inv_map[tup[0]], "text": inv_map[tup[0]]},
            {"rel_id": tup[1], "text": tup[1]},
            {"kb_id": inv_map[tup[2]], "text": inv_map[tup[2]]},
        ])
    return readable_tuples

def _readable_entities_seeds(entities, inv_map):
    readable_entities = []
    try:
        for ent, sc in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent],
                    "pagerank_score": sc})
    except TypeError:
        for ent in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent]})
    return readable_entities

def _readable_entities_corners(entities, inv_map, weight):
    readable_entities = []
    try:
        for ent, sc in entities:
            sc = 0.0
            if inv_map[ent] in weight:
                sc = weight[inv_map[ent]]
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent],
                    "score": sc})

    except TypeError:
        for ent in entities:
            sc = 0.0
            if inv_map[ent] in weight:
                sc = weight[inv_map[ent]]
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent],
                 "score": sc})

    return readable_entities

def _get_answer_coverage(GT, entities, inv_map):
    if len(GT) == 0:
        return -1.
    if len(entities) == 0:
        return 0.
    found, total = 0., 0
    all_entities = set([inv_map[ee] for ee, _ in entities])
    for answer in GT:
        if answer[0] in all_entities: found += 1.
        elif answer[0] + "T00:00:00Z" in all_entities: found += 1.
        total += 1
    return found / total


class SubGraphGenerator:
    def __init__(self, config, property):
        self.config = config
        self.property = property
        self.logger = get_logger(__name__, config)
        self.nerd = self.config["nerd"]
        self.benchmark = self.config["benchmark"]
        self.question_emb_pkl_file = os.path.join(self.config["path_to_data"], self.benchmark, self.config["question_wikiemb_path"])
        self.relation_emb_pkl_file = os.path.join(self.config["path_to_data"], self.benchmark, self.config["relation_wikiemb_path"])
        self.relation_embeddings = pkl.load(open(self.relation_emb_pkl_file, 'rb'))
        self.question_embeddings = pkl.load(open(self.question_emb_pkl_file, 'rb'))

    def generate_subgraph_instance(self, instance, seed_map, bad_questions, ok_questions, answer_recall):
        num_empty_tuples = 0
        evidences = instance["candidate_evidences"]
        QuestionId = instance["Id"]
        QuestionText = instance["Question"]
        GT = get_answer(instance["answers"])
        question_embedding = self.question_embeddings[QuestionId]
        seed_map[QuestionId] = _read_seeds(instance, self.nerd)
        tuples, t_rels, t_ents, line_map, entity_name = get_triple_from_SPO(evidences, self.property)

        if not (tuples):
            entity_map, relation_map, adj_mat = {}, {}, None
        else:
            entity_map, relation_map, adj_mat = _read_facts(tuples, self.relation_embeddings, question_embedding)

        inv_map = {i: k for k, i in entity_map.items()}
        seed_entities = []
        ans_entities = []

        for ee in seed_map[QuestionId]:
            if ee in entity_map:
                seed_entities.append(entity_map[ee])

        if seed_entities:
            for answer in GT:
                if answer[0] in entity_map:
                    ans_entities.append(entity_map[answer[0]])

        if not seed_entities:
            self.logger.info("No seeds found for %s!" % QuestionId)
            extracted_tuples, extracted_ents = [], []
        elif adj_mat is None:
            self.logger.info("No facts for %s!" % QuestionId)
            extracted_tuples, extracted_ents = [], []
        else:
            sd = seed_entities + ans_entities if USEANSWER else seed_entities
            extracted_tuples, extracted_ents = _get_subgraph(
                sd, relation_map, adj_mat)

        if not extracted_tuples:
            num_empty_tuples += 1

        curr_recall = _get_answer_coverage(GT, extracted_ents, inv_map)

        if curr_recall > 0:
            ok_questions.append(QuestionId)
        else:
            bad_questions.append(QuestionId)

        answer_recall += [True if curr_recall > 0 else False]

        all_entities = _readable_entities_seeds(extracted_ents, inv_map)
        entities = _readable_entities_seeds(seed_entities[:MAX_SEEDS], inv_map)
        answers = [{"kb_id": answer[0], "text": answer[1]} for answer in GT]
        tuples = _convert_to_readable(extracted_tuples, inv_map)

        data = {
                "question": QuestionText,
                "entities": entities,
                "answers": answers,
                "id": QuestionId,
                "subgraph": {
                    "entities": all_entities,
                    "tuples": tuples
                },
                "signal": instance["Temporal signal"],
                "type": instance["Temporal question type"],
                "tempentities": list(t_ents),
                "temprelations": list(t_rels),
                "entityname": entity_name,
                "answer_presence": [True if curr_recall > 0 else False],
                "twohopfact_answer_presence": instance["answer_presence"]
                }
        return data, num_empty_tuples

