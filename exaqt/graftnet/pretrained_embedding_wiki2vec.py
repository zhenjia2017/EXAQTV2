"""Script to compute
   relation embeddings for each relation in given list.
   entity embeddings for each entity in given list.
   word embeddings for each word in given word list.
   date embeddings for each date in given list.
   statement_id embeddings for each statement_id."""
import os.path

import numpy as np
import pickle
from nltk.corpus import stopwords as SW
import json
#from wikipedia2vec import Wikipedia2Vec
from nltk.stem import PorterStemmer
PS = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
LC = LancasterStemmer()
from nltk.stem import SnowballStemmer
SB = SnowballStemmer("english")
reference_date = '2021-01-01'
stopwords = set(SW.words("english"))
stopwords.add("'s")
min_date = {"year": -2000, "month": 1, "day": 1}
max_date = {"year": 7000, "month": 12, "day": 31}

word_dim = 100
#tem_dim = 100

word_to_relation = {}
relation_lens = {}

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

def replace_symbols_in_entity(s):
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
    s = s.replace("'", ' ')
    s = s.replace(';', ' ')
    s = s.replace('/', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('+', ' ')
    s = s.replace('.', ' ')
    s = s.strip('"@fr')
    s = s.strip('"@en')
    s = s.strip('"@cs')
    s = s.strip('"@de')
    s = s.strip()
    return s

def _is_date(date):
    is_date = False
    date_range = dict()
    first = ''
    second = ''
    third = ''
    if date.startswith("-") and len(date.split("-")) == 4:
        first = '-' + date.split("-")[1]
        second = date.split("-")[2]
        third = date.split("-")[3]
    if not date.startswith("-") and len(date.split("-")) == 3:
        first = date.split("-")[0]
        second = date.split("-")[1]
        third = date.split("-")[2]
    if is_number(first) and is_number(second) and is_number(third):
        if int(first) <= max_date['year'] and int(first) >= min_date['year'] and int(second) <= 12 and int(
                    second) >= 1 and int(third) <= 31 and int(third) >= 1:
            is_date = True
            date_range['year'] = int(first)
            date_range['month'] = int(second)
            date_range['day'] = int(third)
        if int(third) <= max_date['year'] and int(third) >= min_date['year'] and int(first) <= 12 and int(
                    first) >= 1 and int(second) <= 31 and int(second) >= 1:
            is_date = True
            date_range['year'] = int(third)
            date_range['month'] = int(first)
            date_range['day'] = int(second)
        if int(third) <= max_date['year'] and int(third) >= min_date['year'] and int(second) <= 12 and int(
                    second) >= 1 and int(first) <= 31 and int(first) >= 1:
            is_date = True
            date_range['year'] = int(third)
            date_range['month'] = int(second)
            date_range['day'] = int(first)
    return is_date, date_range


def relationids(relationids_file, relation2id):
    relationid2id = {}
    with open(relationids_file) as json_data:
        relids = json.load(json_data)
        for id in relids:
            relationid2id[id] = relation2id[relids[id]]

def _add_rel_word(word, rid):
    if word not in word_to_relation: word_to_relation[word] = []
    word_to_relation[word].append(rid)
    if rid not in relation_lens: relation_lens[rid] = 0
    relation_lens[rid] += 1

def relation_emb(relations_file, wiki2vec):
    relation2id = load_dict(relations_file)
    reverse_relation2id = {}
    embedding_matrix = []
    no_in_word = []
    for key, value in relation2id.items():
        reverse_relation2id[str(value)] = key
    relation_emb = {r: np.zeros((word_dim,)) for r in reverse_relation2id}
    for i in range(len(reverse_relation2id)):
        i_str = str(i)
        relation = reverse_relation2id[i_str]
        relation = replace_symbols_in_relation(relation)
        relation_li = relation.split()
        #print (relation_li)
        c = 0
        for word in relation_li:
            key_li = [word, word.lower(), PS.stem(word), LC.stem(word), SB.stem(word)]
            #print (key_li)
            flag = 0
            for item in key_li:
                try:
                    value = wiki2vec.get_word_vector(item)
                    flag = 1
                    break
                except KeyError:
                    continue
                    # try:
                    #     value = wiki2vec.get_entity_vector(item)
                    # except KeyError:
                    #     continue
            if flag == 1:
                relation_emb[i_str] += value
                c += 1

        if c > 0:
            relation_emb[i_str] = relation_emb[i_str]/c
        else:
            no_in_word.append(word)
        #if i_str == '351':
        #    print ('351 relation: ',reverse_relation2id[i_str])
        embedding_matrix.append(relation_emb[i_str])

    embedding_matrix = np.asarray(embedding_matrix)
    print("\nword not in dictionary: ", str(len(set(no_in_word))))
    print("\nword not in dictionary: ", set(no_in_word))
    print("\nlen of relations: ", str(len(relation2id)))
    print("\nlen of relation_emb: ", str(len(relation_emb)))
    print("\nlen of embedding_matrix: ", str(len(embedding_matrix)))
    return relation_emb, embedding_matrix

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            #word = line.strip().decode('UTF-8')
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def average_word_emb(word, wordli, wiki2vec, no_in_word):
    vocab_emb = np.zeros((word_dim,))
    flag = 0
    c = 0.0
    for item in wordli:
        try:
            value = wiki2vec.get_word_vector(item)
            flag = 1
            c += 1
        except KeyError:
            continue
            # try:
            #     value = wiki2vec.get_entity_vector(item.capitalize())
            #     flag = 1
            #     c += 1
            # except KeyError:
            #     value = np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
        vocab_emb += value
    if flag == 0:
        no_in_word.append(word)
        return np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
    else:
        return vocab_emb/c

def get_entity_emb_not_in_embdic(entity, wiki2vec):
    ent_li = replace_symbols_in_entity(entity).split()
    ent_emb = np.zeros((word_dim,))
    c = 0.0
    flag = 0
    for item in ent_li:
        item_forms = [item, item.capitalize(), item.lower()]
        for it in item_forms:
            try:
                ent_emb += wiki2vec.get_entity_vector(item)
                c += 1.0
            except KeyError:
                try:
                    ent_emb += wiki2vec.get_word_vector(item)
                    c += 1.0
                except KeyError:
                    continue
            else:
                break
    if c > 0:
        ent_emb = ent_emb/c
        flag = 1
    if flag == 0:
        ent_emb = np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
    return flag, ent_emb

def average_ent_emb(ent, wiki2vec, no_in_entity):
    value = np.random.uniform(low=-0.5, high=0.5, size=(word_dim,))
    ent_forms = [ent, get_upper(ent), ent.strip('"'), ent.strip('"').strip('"@en'), get_upper(ent.strip('"')), get_upper(ent.strip('"').strip('"@en')), ent.lower(), ent.lower().strip('"'), ent.lower().strip('"').strip('"@en')]
    for item in ent_forms:
        try:
            value = wiki2vec.get_entity_vector(item)
        except KeyError:
            try:
                value = wiki2vec.get_word_vector(item)
            except KeyError:
                continue
        else:
            break
    else:
        flag, value = get_entity_emb_not_in_embdic(item, wiki2vec)
        if flag == 0:
            no_in_entity.append(ent)

    return value

def word_emb(vocab_file, wiki2vec):
    word2id = load_dict(vocab_file)
    reverse_word2id = {}
    embedding_matrix = []
    no_in_word = []
    for key, value in word2id.items():
        reverse_word2id[str(value)] = key
    vocab_emb = {r: np.random.uniform(low=-0.5, high=0.5, size=(word_dim,)) for r in reverse_word2id}
    for i in range(len(reverse_word2id)):
        i_str = str(i)
        key = reverse_word2id[i_str]
        key_li = [key, key.capitalize(), PS.stem(key), LC.stem(key), SB.stem(key)]
        for item in key_li:
            try:
                vocab_emb[i_str] = wiki2vec.get_word_vector(item)
            except KeyError:
                continue
                # try:
                #     vocab_emb[i_str] = wiki2vec.get_entity_vector(item)
                # except KeyError:
                #     continue
            else:
                break
        else:
            if '-' in key or '/' in key:
                if '-' in key:
                    vocab_emb[i_str] = average_word_emb(key, key.split('-'), wiki2vec, no_in_word)
                else:
                    vocab_emb[i_str] = average_word_emb(key, key.split('/'), wiki2vec, no_in_word)
            else:
                no_in_word.append(key)

        embedding_matrix.append(vocab_emb[i_str])

    embedding_matrix = np.asarray(embedding_matrix)
    print ("\nword not in dictionary: ", str(len(no_in_word)))
    print("\nword not in dictionary: ", set(no_in_word))
    print ("\nlen of words: ", str(len(word2id)))
    print("\nlen of vocab_emb: ", str(len(vocab_emb)))
    print("\nlen of embedding_matrix: ", str(len(embedding_matrix)))
    return vocab_emb, embedding_matrix

# def _get_statement_tememb(sta_ent_rel, date2id, entity_name):
#     entities = list(sta_ent_rel["ent"])
#     tem_emb = np.zeros((word_dim,))
#     flag = 0
#     c = 0
#     for entity in entities:
#         entity_label = entity_name[entity]
#         if entity_label in date2id:
#
#             tem_emb += _get_tem_emb(entity_label)
#             flag = 1
#             c += 1
#     if flag == 1: return tem_emb/c
#     else:
#         return _get_tem_emb(reference_date)

def get_upper(words):
    newwords = []
    for i in range(0,len(words)):
        if i ==0 :
            newwords.append(words[i].upper())
        elif words[i - 1] == ' ':
            newwords.append(words[i].upper())
        else:
            newwords.append(words[i])
    upper = ("").join(newwords)
    return upper

def entity_emb(entity_file, date_file, entity_name_file, wiki2vec):
    entity2id = load_dict(entity_file)
    date2id = load_dict(date_file)
    with open(entity_name_file, 'rb') as fin:
        entity_name = pickle.load(fin)

    reverse_entity2id = {}
    embedding_matrix = []
    no_in_entity = []
    for key, value in entity2id.items():
        if key in entity_name:
            reverse_entity2id[str(value)] = entity_name[key]
        else:
            reverse_entity2id[str(value)] = key

    entity_emb = {r: np.zeros((word_dim,)) for r in reverse_entity2id}
    for i in range(len(reverse_entity2id)):
        i_str = str(i)
        entity_label = reverse_entity2id[i_str].strip('"')
        if entity_label in date2id:
            try:
                entity_emb[i_str] = wiki2vec.get_word_vector(entity_label)
            except KeyError:
                if '-' in entity_label or '/' in entity_label:
                    if '-' in entity_label:
                        entity_emb[i_str] = average_word_emb(entity_label, entity_label.split('-'), wiki2vec, no_in_entity)
                    else:
                        entity_emb[i_str] = average_word_emb(entity_label, entity_label.split('/'), wiki2vec, no_in_entity)

        else:
            entity_emb[i_str] = average_ent_emb(entity_label,wiki2vec,no_in_entity)

        embedding_matrix.append(entity_emb[i_str])
    embedding_matrix = np.asarray(embedding_matrix)
    print("entity not in embedding dic: ", str(len(set(no_in_entity))))
    print("len of entity_emb: ", str(len(entity_emb)))
    print("len of embedding_matrix: ", str(len(embedding_matrix)))
    return entity_emb, embedding_matrix






