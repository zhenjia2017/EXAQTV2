from tqdm import tqdm
import json
import numpy as np

data_folder = '/GW/qa/work/exact/baseline/GraftNet/GraftNet-master/datasets/webqsp/full/'
train_data = 'train.json'
test_data = 'test.json'
data_file = data_folder + train_data
data_file = data_folder + test_data
data = []
num_kb_relation = 1000
max_facts = 0
count = 0
with open(data_file) as f_in:
    for line in tqdm(f_in):
        line = json.loads(line)
        data.append(line)
        #max_relevant_doc = max(max_relevant_doc, len(line['passages']))
        if count < 2 :
            print (line['subgraph']['tuples'])
            print (line.keys())
            print (line['subgraph'].keys())
            print (line['question'])
            count += 1
        max_facts = max(max_facts, 2 * len(line['subgraph']['tuples']))
#print('max_relevant_doc: ', max_relevant_doc)
print('max_facts: ', max_facts)
num_data = len(data)
batches = np.arange(num_data)
kb_adj_mats = np.empty(num_data, dtype=object)
kb_fact_rels = np.full((num_data, max_facts), num_kb_relation, dtype=int)
print('num_data: ', num_data)
print('kb_adj_mats: ', kb_adj_mats)
print('kb_fact_rels: ', kb_fact_rels)

# file = r'G:\QA\GraftNet-master\preprocessing\scratch\stagg_linked_questions.txt'
# num_lines = sum(1 for line in open(file,'r'))
# with open(file,'r') as f:
#     for line in tqdm(f, total=num_lines):
#         print(line)