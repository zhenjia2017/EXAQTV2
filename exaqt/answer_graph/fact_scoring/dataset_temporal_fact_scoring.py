import truecase
import json
import torch
import random
from tqdm import tqdm
from sklearn import model_selection
import csv
import pandas as pd
from exaqt.evaluation import answer_presence, evidence_has_answer
from exaqt.answer_graph.fact_scoring.dataset_fact_scoring import BERTDataset
from exaqt.answer_graph.fact_retriever.fact_er import FactRetriever
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
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

class DatasetTemporalFactScoring(torch.utils.data.Dataset):
    def __init__(self, config, property):
        self.config = config
        self.property = property
        self.nerd = self.config["nerd"]
        self.fact_retriever = FactRetriever(config, property)
        # initialize clocq for KB-facts
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

    def load_data_csv(self, train_path, dev_path, output_path):
        merged_df = pd.DataFrame()
        for train_path in [train_path, dev_path]:
            dfx = pd.read_csv(train_path)
            merged_df = pd.concat([merged_df, dfx], ignore_index=True)
        merged_df.to_csv(output_path, index=False)

        dfx = pd.read_csv(output_path).fillna("none")
        df_train, df_valid = model_selection.train_test_split(dfx, test_size=0.2, random_state=42,
                                                              stratify=dfx.label.values)
        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        train_dataset = BERTDataset(df_train.question.values, df_train.context.values, df_train.label.values,
                                    self.config)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, num_workers=4)

        valid_dataset = BERTDataset(df_valid.question.values, df_valid.context.values, df_valid.label.values,
                                    self.config)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=50, num_workers=1)

        print("Data transformed")
        return train_data_loader, valid_data_loader, df_train

    def load_data(self, train_path, dev_path, output_path):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.
        """
        #print(f"Start with {train_path} {dev_path}")

        # load parameters
        max_pos_examples = self.config["temporal_fs_max_pos_evidences"]
        max_neg_examples = self.config["temporal_fs_max_neg_evidences"]

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
            evidences = instance["temporal_evidences"]
            if len(evidences) == 0:
                continue
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
            evi_answers = instance['answers']
            for ans in answers:
                if 'WikidataQid' in ans:
                    Qid = ans['WikidataQid'].lower()
                    GT.append(Qid)
                if "AnswerArgument" in ans:
                    GT.append(ans['AnswerArgument'].replace('T00:00:00Z', '').lower())

            for evidence in evidences:
                for item in evidence["statement_spo_list"]:
                    f_list.append(item)

            ans_pres, _ = answer_presence(evidences, evi_answers)
            print("\nhop1 temporal evidences answer presence", ans_pres)
            if ans_pres > 0:
                have_answer_spo_list = f_list
            else:
                wiki_ids = instance["elq"]
                if self.nerd == "elq-wat":
                    wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
                elif self.nerd == "elq-tagme":
                    wiki_ids += instance["tagme"]
                elif self.nerd == "elq-tagme-wat":
                    wiki_ids += instance["wat"]
                    wiki_ids += instance["tagme"]
                # skip instances for which no answer was found

                if not ans_pres:
                    wiki_ids = list(set([item[0] for item in wiki_ids]))
                    hop2_temporal_evidences = self.fact_retriever.two_hop_temporal_fact_retriever(wiki_ids)
                    hop2_ans_pres, _ = answer_presence(hop2_temporal_evidences, evi_answers)
                    print("\nhop2 temporal evidences answer presence", hop2_ans_pres)
                    if not hop2_ans_pres:
                        continue
                    else:
                        tf_list = []
                        for evidence in hop2_temporal_evidences:
                            for item in evidence["statement_spo_list"]:
                                tf_list.append(item)
                        have_answer_spo_list = tf_list


            for line in have_answer_spo_list:
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


