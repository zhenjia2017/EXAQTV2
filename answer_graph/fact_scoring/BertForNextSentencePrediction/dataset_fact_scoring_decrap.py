import time
import json
import torch
import random
from tqdm import tqdm

from exaqt.evaluation import answer_presence, evidence_has_answer

# set seed for reproducability
random.seed(7)

def format_input(question, evidence):
    fact_text = " ".join([item['label'] for item in evidence["fact"]])
    return f"{question}[SEP]{fact_text}"

class DatasetFactScoring(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer

        # load data
        input_encodings, labels = self._load_data(path)
        self.input_encodings = input_encodings
        self.labels = labels
        print(f"Done with {path}")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        labels = self.labels[idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }
        return item

    def __len__(self):
        return len(self.labels)

    def _load_data(self, path):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.
        """
        print(f"Start with {path}")

        # load parameters
        max_pos_examples = self.config["fs_max_pos_evidences"]
        max_neg_evidences = self.config["fs_max_neg_evidences"]

        # initialize
        inputs = list()
        labels = list()

        # open data
        print("Transform data")
        with open(path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                # transform data to fit model input format
                question = instance["Question"]
                answers = instance["answers"]
                evidences = instance["candidate_evidences"]

                # skip instances for which no answer was found
                ans_pres = answer_presence(evidences, answers)
                if not ans_pres:
                    continue

                # sampling
                pos_sample = list()
                neg_sample = list()

                all_evidences = instance["candidate_evidences"]
                random.shuffle(all_evidences)
                for evidence in all_evidences:
                    if evidence_has_answer(evidence, answers):
                        pos_sample.append(evidence)
                    elif len(neg_sample) < max_neg_evidences:
                        neg_sample.append(evidence)

                if len(pos_sample) > max_pos_examples:
                    continue

                # add positive samples
                for evidence in pos_sample:
                    input_ = format_input(question, evidence)
                    inputs.append(input_)
                    labels.append(1)  # positive example would be given label 1

                # add negative samples
                for evidence in neg_sample:
                    input_ = format_input(question, evidence)
                    inputs.append(input_)
                    labels.append(0)  # negative example would be given label 0

        print("Data transformed")

        # encode
        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["fs_max_input_length"]
        )
        print("Data encoded")

        return input_encodings, labels
