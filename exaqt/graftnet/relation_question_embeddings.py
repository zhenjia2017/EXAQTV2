"""Script to compute relation embeddings for each relation in given list."""

import numpy as np
import pickle as pkl
from pathlib import Path
from tqdm import tqdm
import json
from wikipedia2vec import Wikipedia2Vec
import os

from exaqt.library.utils import get_logger

word_to_relation = {}
relation_lens = {}
word_to_question = {}
question_lens = {}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def gloveEmbedding(embedding_filepath):
    # glove_dict = dict()
    # glove_emd_matrix = list()
    all_word_embedding = dict()

    with open(embedding_filepath) as fin:
        for line in tqdm(fin):
            #word, vec = line.strip().split(None, 1)
            if line.strip():
                seg_res = line.split(" ")
                seg_res = [word.strip() for word in seg_res if word.strip()]

                key = seg_res[0]
                value = [float(word) for word in seg_res[1:]]
                all_word_embedding[key] = value
    # return all_word_embedding
    # glove_dict['#UNK#'] = len(glove_dict)
    # for word in all_word_embedding:
    #     glove_dict[word] = len(glove_dict)
    #
    # glove_emd_matrix.append(np.random.normal(size=(300, )).tolist())
    # for word in all_word_embedding:
    #     glove_emd_matrix.append(all_word_embedding[word])

    return all_word_embedding

def relationids(relationids_file, relation2id):
    relationid2id = {}
    with open(relationids_file) as json_data:
        relids = json.load(json_data)
        for id in relids:
            relationid2id[id] = relation2id[relids[id]]

def _add_ques_word(word, v):
    if word not in word_to_question: word_to_question[word] = []
    word_to_question[word].append(v)
    if v not in question_lens: question_lens[v] = 0
    question_lens[v] += 1

def _add_rel_word(word, v):
    if word not in word_to_relation: word_to_relation[word] = []
    word_to_relation[word].append(v)
    if v not in relation_lens: relation_lens[v] = 0
    relation_lens[v] += 1

def relation_emb_f(relations, wiki2vec, word_dim):
    for qid in relations:
        one_relation = relations[qid]["label"]
        for word in one_relation.split():
            _add_rel_word(word, one_relation)

    relation_emb = {r: np.zeros((word_dim,)) for r in relation_lens}

    for word in word_to_relation:
        try:
            vec = wiki2vec.get_word_vector(word)
            for qid in word_to_relation[word]:
                relation_emb[qid] += vec
        except KeyError:
            continue

    for relation in relation_emb:
        relation_emb[relation] = relation_emb[relation] / relation_lens[relation]
    return relation_emb

def question_emb_f(instances, wiki2vec, dim):
    for instance in instances:
        question_id, question_text = instance["Id"], instance["Question"]
        for word in question_text.split():
            _add_ques_word(word, question_id)

    question_emb = {r: np.zeros((dim,)) for r in question_lens}
    for word in word_to_question:
        try:
            vec = wiki2vec.get_word_vector(word)
            for v in word_to_question[word]:
                question_emb[v] += vec
        except KeyError:
            continue

    for v in question_emb:
        question_emb[v] = question_emb[v] / question_lens[v]
    return question_emb

def relation_emb(relations, wiki2vec, word_dim):
    for qid in relations:
        one_relation = relations[qid]["label"]
        for word in one_relation.split():
            _add_rel_word(word, one_relation)

    relation_emb = {r: np.zeros((word_dim, )) for r in relation_lens}

    for word in word_to_relation:
        try:
            vec = wiki2vec.get_word_vector(word)
            for qid in word_to_relation[word]:
                relation_emb[qid] += vec
        except KeyError:
            continue

    for relation in relation_emb:
        relation_emb[relation] = relation_emb[relation] / relation_lens[relation]

    return relation_emb

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

class RelationQuestionEmbedding:
    def __init__(self, config, property):
        self.config = config
        self.property = property
        self.logger = get_logger(__name__, config)
        self.model_file = os.path.join(self.config["path_to_data"], self.config["wikipedia2vec_path"])
        self.wiki2vec = Wikipedia2Vec.load(self.model_file)
        self.word_dim = self.config["word_dim"]
        self.benchmark = self.config["benchmark"]
        self.benchmark_path = self.config["benchmark_path"]
        self.generate_embeddings()

    def generate_embeddings(self):
        test_input_path = os.path.join(self.benchmark_path, self.benchmark, self.config["test_input_path"])
        dev_input_path = os.path.join(self.benchmark_path, self.benchmark, self.config["dev_input_path"])
        train_input_path = os.path.join(self.benchmark_path, self.benchmark, self.config["train_input_path"])
        instances = []
        for file in [test_input_path, dev_input_path, train_input_path]:
            data = json.load(open(file))
            instances += data

        question_emb_pkl_file = os.path.join(self.config["path_to_data"], self.benchmark, self.config["question_wikiemb_path"])
        #Path(question_emb_pkl_file).mkdir(parents=True, exist_ok=True)
        self.logger.info("Generate Question Embeddings")
        question_embeddings = question_emb_f(instances, self.wiki2vec, self.word_dim)
        pkl.dump(question_embeddings, open(question_emb_pkl_file, "wb"))
        self.logger.info("Save Question Embeddings File")

        self.logger.info("Generate Relation Embeddings")
        relation_emb_pkl_file = os.path.join(self.config["path_to_data"], self.benchmark, self.config["relation_wikiemb_path"])
        #Path(relation_emb_pkl_file).mkdir(parents=True, exist_ok=True)
        relation_embeddings = relation_emb_f(self.property, self.wiki2vec, self.word_dim)
        pkl.dump(relation_embeddings, open(relation_emb_pkl_file, "wb"))
        self.logger.info("Save Relation Embeddings File")


