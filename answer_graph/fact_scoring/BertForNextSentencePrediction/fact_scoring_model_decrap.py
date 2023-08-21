"""
Module to predict the relevance of a KB fact for an AR.
Was used in preliminary experiments for meeting on 2021-10-12,
to show that ARs are indeed self-sufficient.

Input: relevant turn, question, KB fact
Output: Relevance score
"""
import os
import torch
import transformers

from exaqt.answer_graph.fact_scoring.BertForNextSentencePrediction.custom_trainer import CustomTrainer
import exaqt.answer_graph.fact_scoring.dataset_fact_scoring as dataset
from exaqt.library.utils import get_logger

class FactScoringModel(torch.nn.Module):
    def __init__(self, config):
        super(FactScoringModel, self).__init__()
        self.model = transformers.BertForNextSentencePrediction.from_pretrained("bert-base-cased")
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

        self.config = config
        self.logger = get_logger(__name__, config)
        self.logger.info("Use pytorch device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

        # define model path
        #self.model_path = f"fact_scoring/model"
        self.model_path = self.config["fs_model_save_path"]

    def inference(self, input_texts):
        """Run the model on the given input."""
        # encode inputs
        input_encodings = self.tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=self.config["fs_max_input_length"],
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            input_encodings = input_encodings.to("cuda")

        # inference
        outputs = self.model(**input_encodings)
        scores = outputs.logits.detach()
        return scores

    def set_eval_mode(self):
        """Set model to eval mode."""
        self.model.eval()

    def load(self):
        """Load model."""
        path = os.path.join(self.model_path, "pytorch_model.bin")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, train_path, dev_path):
        """Train model."""
        # load datasets
        train_dataset = dataset.DatasetFactScoring(self.config, self.tokenizer, train_path)
        dev_dataset = dataset.DatasetFactScoring(self.config, self.tokenizer, dev_path)


        print("Training arguments")
        # arguments for training
        training_args = transformers.TrainingArguments(
            output_dir="fact_scoring/results",  # output directory
            num_train_epochs=5,  # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="fact_scoring/logs",  # directory for storing logs
            logging_steps=10000,
            evaluation_strategy="epoch",
        )
        self.logger.info( f"Training arguments: {training_args}")
        self.logger.info(f"Custom trainer")

        # create the object for training
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            path_to_best_model=self.model_path,
        )
        self.logger.info(f"Start training")
        # training progress
        trainer.train()
