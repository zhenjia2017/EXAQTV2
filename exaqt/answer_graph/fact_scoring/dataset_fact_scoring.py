import truecase
import json
import torch
import random
import os
from tqdm import tqdm
from sklearn import model_selection
import csv
import pandas as pd
import transformers

# set seed for reproducability
random.seed(7)

def format_input(question, evidence, property):
    fact_text = ''
    question = truecase.get_true_case(question)
    for line in evidence["statement_spo_list"]:
        triple = line.strip().split('||')
        sub_name = triple[2].replace('T00:00:00Z', '')
        obj_name = triple[6].replace('T00:00:00Z', '')
        rel_name = triple[4]
        if triple[4] in property:
            rel_name = property[triple[4]]['label']
        if "-ps:" in triple[0]:
            fact_text += sub_name + ' ' + rel_name + ' ' + obj_name
        elif "-pq:" in triple[0]:
            fact_text += ' and' + ' ' + rel_name + ' ' + obj_name

    return question, fact_text

class BERTDataset:
    def __init__(self, q1, q2, target, config):
        self.q1 = q1
        self.q2 = q2
        self.target = target
        self.bert_model_path = os.path.join(config["path_to_data"], config["bert_model_path"])
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_path, do_lower_case = False)
        self.max_len = 512

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

class DatasetFactScoring(torch.utils.data.Dataset):
    def __init__(self, config, property):
        self.config = config
        self.property = property

    def load_data(self, train_path, dev_path, output_path):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.
        """
        print(f"Start with {train_path} {dev_path}")

        # load parameters
        max_pos_examples = self.config["fs_max_pos_evidences"]
        max_neg_examples = self.config["fs_max_neg_evidences"]

        train_set_instances = []
        train_list = []
        count = 0
        # open data
        print("Transform data")
        with open(train_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                train_set_instances.append(instance)

        with open(dev_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                train_set_instances.append(instance)
                # transform data to fit model input format

        for instance in train_set_instances:
            question_text = instance["Question"]
            evidences = instance["candidate_evidences"]
            f_list = []
            spo_sta = {}
            ans_sta = {}
            main_rel_sta = {}
            GT = []
            positive = []
            pos_rel = []
            negative = []
            ques_id = str(instance['Id'])
            answers = instance['Answer']
            for ans in answers:
                if 'WikidataQid' in ans:
                    Qid = ans['WikidataQid'].lower()
                    GT.append(Qid)
                if "AnswerArgument" in ans:
                    GT.append(ans['AnswerArgument'].replace('T00:00:00Z', '').lower())
            for evidence in evidences:
                for item in evidence["statement_spo_list"]:
                    f_list.append(item)

            for line in f_list:
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
                if triple[4] in self.property:
                    rel_name = self.property[triple[4]]['label']
                if triple[3] in self.property:
                    main_rel_name = self.property[triple[3]]['label']
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
                    if ent in GT:
                        pos_flag = 1
                        break
                if pos_flag == 1:
                    positive.append(statement_id)
                    pos_rel.append(main_rel_sta[statement_id])

            for statement_id in ans_sta:
                pos_flag = 0
                for ent in ans_sta[statement_id]:
                    if len(GT) > 0 and ent in GT:
                        pos_flag = 1
                        break
                if pos_flag == 0 and main_rel_sta[statement_id] not in pos_rel:
                    negative.append(statement_id)

            if len(positive) > 0 and len(negative) > 0:
                count += 1
                positive_train = random.sample(positive, max_pos_examples)
                if len(negative) < 5:
                    negative_train = random.sample(negative, len(negative))
                else:
                    negative_train = random.sample(negative, max_neg_examples)
                train_id = 'train_' + ques_id
                train_ques_text = question_text
                train_ques_text = truecase.get_true_case(train_ques_text)
                context = " ".join(spo_sta[positive_train[0]]['ps']) + " and".join(spo_sta[positive_train[0]]['pq'])
                train_list.append([train_id, train_ques_text, context, 1])
                for neg_sta in negative_train:
                    context = (" ").join(spo_sta[neg_sta]['ps']) + (" and").join(spo_sta[neg_sta]['pq'])
                    if [train_id, train_ques_text, context, 0] not in train_list:
                        train_list.append([train_id, train_ques_text, context, 0])

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["questionId", "question", "context", "label"])
            for sample in train_list:
                writer.writerow(sample)

        dfx = pd.read_csv(output_path).fillna("none")
        df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.2, random_state=42,
                                                              stratify=dfx.label.values)
        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        train_dataset = BERTDataset(df_train.question.values, df_train.context.values, df_train.label.values, self.config)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, num_workers=4)

        valid_dataset = BERTDataset(df_valid.question.values, df_valid.context.values, df_valid.label.values, self.config)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=50, num_workers=1)

        print("Data transformed")
        return train_data_loader, valid_data_loader, df_train


