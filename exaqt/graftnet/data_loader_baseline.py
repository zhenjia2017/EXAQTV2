import json
import numpy as np
from tqdm import tqdm

def replace_symbols(s):
    #s = s.replace('<entity>', ' ')
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
    s = s.replace('\'s',' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n',' ')
    s = s.strip(',')
    s = s.strip('.')
    s = s.strip('#')
    s = s.strip('-')
    s = s.strip('\'')
    s = s.strip(';')
    s = s.strip('\"')
    s = s.strip('/')
    s = s.rstrip('?')
    s = s.rstrip('!')
    s = s.strip()
    return s

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

class DataLoader():
    def __init__(self, data_file, word2id, relation2id, entity2id, max_query_word):
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.num_kb_relation = len(relation2id)

        print(('loading data from', data_file))
        self.data = []

        with open(data_file, encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                try:
                    line = json.loads(line)
                except:
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                #self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['cliquetuples']))

        print(('max_facts: ', self.max_facts))
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i:entity for entity, i in list(entity2id.items())}
        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()
        print(('max_local_entity', self.max_local_entity))
        print('preparing data ...')
        self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), len(self.relation2id), dtype=int)
        self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)


        ###date entity in KB
        self.max_dates = 0


        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        for sample in tqdm(self.data):
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            for global_entity, local_entity in list(g2l.items()):
                if local_entity != 0: # skip question node
                    self.local_entities[next_id, local_entity] = global_entity
                    # get a list of local entities

            entity2fact_e, entity2fact_f = [], []
            fact2entity_f, fact2entity_e = [], []

            # relations in local KB
            #spo graph
            for i, tpl in enumerate(sample['subgraph']['tuples']):
            #clique graph
            #for i, tpl in enumerate(sample['subgraph']['cliquetuples']):
                sbj, rel, obj = tpl
                if sbj['text'] in self.entity2id and obj['text'] in self.entity2id:
                    entity2fact_e += [g2l[self.entity2id[sbj['text']]]]
                    entity2fact_f += [i]
                    fact2entity_f += [i]
                    fact2entity_e += [g2l[self.entity2id[obj['text']]]]
                    self.kb_fact_rels[next_id, i] = self.relation2id[rel['text']]

            # build connection between question and entities in it
            for j, entity in enumerate(sample['entities']):
            #{"text": inv_map[ent], "kb_id": inv_map[ent],"score": sc})
                num_seed = len(sample['entities'])
                if entity['text'] in self.entity2id:
                    self.q2e_adj_mats[next_id, g2l[self.entity2id[str(entity['text'])]], 0] = 1.0 / num_seed

            # tokenize question
            ques_words = [replace_symbols(item) for item in sample['question'].strip().split()]
            count_query_length[len(ques_words)] += 1

            for j, word in enumerate(ques_words):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    #else:
                    #    self.query_texts[next_id, j] = self.word2id['__unk__']



            # construct distribution for answers
            for answer in sample['answers']:
                if answer['kb_id'] in self.entity2id:
                    if self.entity2id[answer["kb_id"]] in g2l:
                        self.answer_dists[next_id, g2l[self.entity2id[answer["kb_id"]]]] = 1.0

            self.kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int), np.array(entity2fact_e, dtype=int), np.array([1.0] * len(entity2fact_f))), (np.array(fact2entity_e, dtype=int), np.array(fact2entity_f, dtype=int), np.array([1.0] * len(fact2entity_e)))

            next_id += 1

    def _build_kb_adj_mat(self, sample_ids, fact_dropout):
        """Create sparse matrix representation for batched data"""
        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        for i, sample_id in enumerate(sample_ids):
            (mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1) = self.kb_adj_mats[sample_id]
            assert len(val0) == len(val1)
            num_fact = len(val0)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[ : num_keep_fact]
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(len(mask_index), i, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0[mask_index])
            mats0_1 = np.append(mats0_1, mat0_1[mask_index])
            vals0 = np.append(vals0, val0[mask_index])
            # mat1
            mats1_batch = np.append(mats1_batch, np.full(len(mask_index), i, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0[mask_index])
            mats1_1 = np.append(mats1_1, mat1_1[mask_index])
            vals1 = np.append(vals1, val1[mask_index])

        return (mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
        #print ('\n\nself.query_texts[sample_ids]',self.query_texts[sample_ids])
        return self.local_entities[sample_ids], \
               self.q2e_adj_mats[sample_ids], \
               (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
               self.kb_fact_rels[sample_ids], \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)
            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print(('avg local entity: ', total_local_entity / next_id))
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):

        for entity in entities:
            entity_text = entity['text']
            #entity_text = entity["kb_id"]
            if entity_text in entity2id:
                entity_global_id = entity2id[entity_text]
                if entity_global_id not in g2l:
                    g2l[entity2id[entity_text]] = len(g2l)

            else:
                print ("\n\nentity not in entity2id:")
                print(entity_text)