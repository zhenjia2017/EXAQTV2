"""
Module to score facts for a question.

Input: question, relevant KB facts
Output: Relevance score
"""
import torch
from pathlib import Path

MSG = "Use pytorch device: {}".format("cuda" if torch.cuda.is_available() else "cpu")
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from exaqt.answer_graph.fact_scoring.engine import train_fn, eval_fn
from sklearn import metrics
import os
from exaqt.library.utils import get_config, get_logger

class BERTBaseUncased(torch.nn.Module):
    def __init__(self, config):
        super(BERTBaseUncased, self).__init__()
        self.bert_model_path = os.path.join(self.config["path_to_data"], self.config["bert_model_path"])
        self.model = transformers.BertModel.from_pretrained(self.bert_model_path)
        self.bert_drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        outs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(outs.pooler_output)
        output = self.out(bo)
        return output

class FactScoringModel(torch.nn.Module):
    def __init__(self, config, best_model_path):
        super(FactScoringModel, self).__init__()
        self.config = config
        self.logger = get_logger(__name__, config)
        self.logger.info(MSG)

        self.best_model_path = best_model_path
        self.model = BERTBaseUncased(config)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

        self.bert_model_path = os.path.join(self.config["path_to_data"], self.config["bert_model_path"])
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_path, do_lower_case=False)

    def train(self, train_data_loader, valid_data_loader, df_train):
        param_optimizer = list(self.model.named_parameters())
        device = torch.device("cuda")
        no_decay = ['bias', 'LayerNorm.bias', 'LayetNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.1}]

        num_train_steps = int(len(df_train) / 50 * 2)
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
        print("\nnum_train_steps:")
        print(num_train_steps)
        self.model = torch.nn.DataParallel(self.model)
        best_accuracy = 0
        for epoch in range(2):
            train_fn(train_data_loader, self.model, optimizer, device, scheduler)
            outputs, targets = eval_fn(valid_data_loader, self.model, device)

            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            print("Accuracy Score = ", accuracy)

            if accuracy > best_accuracy:
                print("Saving Checkpoint!!!")
                model_dir = os.path.dirname(self.best_model_path)
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.best_model_path)
                best_accuracy = accuracy

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def load(self):
        """Load model."""
        print ("best_model_path")
        print (self.best_model_path)
        best_model = torch.load(self.best_model_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in best_model.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.set_eval_mode()
        print("load model successfully!")

    def inference(self, question, fact):
        device = torch.device("cuda")

        inputs = self.tokenizer.encode_plus(
            question,
            fact,
            add_special_tokens=True,
            max_length=512,
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

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        return outputs[0][0]
