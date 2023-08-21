"""
Script to train classifier through fine-tuning bert.
Script to get score for each fact (or sentence) using the model.
"""
import truecase
import os
import json
import pickle
import globals
import transformers
#import dataset
import torch
import pandas as pd
from sklearn import model_selection
import engine
import numpy as np
#from model import BERTBaseUncased
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
import torch.nn as nn

DEVICE = "cuda"
MAX_LEN =  512
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 50
EPOCHS = 2
ACCUMULATION = 4

#train question_relevant_fact_selection_model and score fact

cfg = globals.get_config(globals.config_file)
BERT_PATH = cfg["model_path"] + "bert_base_cased/"
MODEL_PATH = cfg["model_path"] + "phase1_model_new.bin"
TRAINING_FILE = cfg["data_path"] + "phase1_relevant_fact_selection_trainset_new.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case = False)

class BERTDataset:
    def __init__(self, q1, q2, target):
        self.q1 = q1
        self.q2 = q2
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, item):
        q1 = str(self.q1[item])
        q2 = str(self.q2[item])

        q1 = " ".join(q1.split())
        q2 = " ".join(q2.split())

        inputs = self.tokenizer.encode_plus(q1, q2, add_special_tokens=True, max_length=self.max_len, padding='longest',
                                            truncation=True)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {'ids': torch.tensor(ids, dtype=torch.long), 'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.target[item], dtype=torch.long)}


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        outs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(outs.pooler_output)
        output = self.out(bo)
        return output

def ques_tuple_prediction(ques,tuple,model):
    tokenizer = TOKENIZER
    max_len = MAX_LEN
    device = torch.device("cuda")
    inputs = tokenizer.encode_plus(
        ques,
        tuple,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt')

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    ids = ids.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

def get_statement_fact(spo_file,pro_info):
    spo_sta = {}
    with open(spo_file) as f:
        for line in f:
            triple = line.strip().split('||')
            if len(triple) < 7 or len(triple) > 7: continue
            statement_id = line.strip().split("||")[0].replace("-ps:", "").replace("-pq:", "").lower()
            sub_name = triple[2].replace('T00:00:00Z', '')
            obj_name = triple[6].replace('T00:00:00Z', '')
            rel_name = triple[4]
            if triple[4] in pro_info:
                rel_name = pro_info[triple[4]]['label']
            if statement_id not in spo_sta:
                spo_sta[statement_id] = dict()
                spo_sta[statement_id]['ps'] = []
                spo_sta[statement_id]['pq'] = []

            if "-ps:" in line.split("||")[0]:
                spo_sta[statement_id]['ps'].append(sub_name + ' ' + rel_name + ' ' + obj_name)
            if "-pq:" in line.split("||")[0]:
                spo_sta[statement_id]['pq'].append(' ' + rel_name + ' ' + obj_name)

    return spo_sta

def run():
    print(TRAINING_FILE)
    print(EPOCHS)
    dfx = pd.read_csv(TRAINING_FILE).fillna("none")
    df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.2, random_state=42,
                                                          stratify=dfx.label.values)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    # pass the q1,q2,target here .
    train_dataset = BERTDataset(df_train.question.values, df_train.context.values, target=df_train.label.values)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4)

    valid_dataset = BERTDataset(df_valid.question.values, df_valid.context.values, df_valid.label.values)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=1)

    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayetNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.1}]

    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
    print("\nnum_train_steps:")
    print(num_train_steps)
    model = nn.DataParallel(model)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print("Accuracy Score = ", accuracy)

        if accuracy > best_accuracy:
            print("Saving Checkpoint!!!")
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str, default='test')
    argparser.add_argument('-s', '--start', type=int, default=0)
    argparser.add_argument('-e', '--end', type=int, default=4000)
    argparser.add_argument('-p', '--predicate', type=int, default=0)
    argparser.add_argument('-t', '--train', type=int, default=0)

    args = argparser.parse_args()
    predicate = args.predicate
    train = args.train
    if train == 1:
        print("Training Running")
        run()
    if predicate == 1:
        print("Predicting Running")
        dataset = args.dataset
        start = args.start
        end = args.end

        device = torch.device("cuda")
        MODEL = BERTBaseUncased()
        model = torch.load(MODEL_PATH)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        MODEL.load_state_dict(new_state_dict)
        MODEL.to(device)
        MODEL.eval()
        print ("load model successfully!")
        pro_info = globals.ReadProperty.init_from_config().property
        test = cfg["data_path"] + cfg["test_data"]
        dev = cfg["data_path"] + cfg["dev_data"]
        train = cfg["data_path"] + cfg["train_data"]
        if dataset == 'dev':
            in_file = dev
        elif dataset == 'test':
            in_file = test
        elif dataset == 'train':
            in_file = train
        no_spo_file_count = 0
        data = json.load(open(in_file))
        for question in data:
            sta_score = []
            ques_id = question["Id"]
            if data.index(question) in range(start, end):
                ques_text = question["Question"]
                path = cfg['ques_path'] + 'ques_' + str(question["Id"])
                spo_file = path + '/SPO_new.txt'
                spo_score_file = path + '/SPO_score_new.txt'
                spo_rank_file = path + '/SPO_rank_new.pkl'
                if os.path.exists(spo_score_file):
                    print ("spo_score_file exist!")
                    continue
                if not (os.path.exists(spo_file)):
                    no_spo_file_count += 1
                    continue
                ques_text = truecase.get_true_case(ques_text)
                spo_sta = get_statement_fact(spo_file, pro_info)
                for statement_id in spo_sta:
                    #context = " ".join(spo_sta[statement_id]['ps']) + " and".join(spo_sta[statement_id]['pq'])
                    context = " ".join(spo_sta[statement_id]['ps']) + " and".join(spo_sta[statement_id]['pq'])
                    score = ques_tuple_prediction(ques_text, context, MODEL)
                    sta_score.append((statement_id, float(score), context))
                sta_score = sorted(sta_score, key=lambda tup: tup[1], reverse=True)
                f1 = open(spo_score_file, 'w', encoding='utf-8')
                rank_fact_dic = {}
                for i, item in enumerate(sta_score):
                    rank_fact_dic[str(i+1)] = str(item[0]) +  '\t' +  str(item[1])
                    f1.write(str(i+1) + '\t' + str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')
                f1.close()
                f3 = open(spo_rank_file, 'wb')
                pickle.dump(rank_fact_dic, f3)
                f3.close()
        print("\nno_spo_file_count: ", str(no_spo_file_count))

