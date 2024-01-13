import os.path
import torch
from exaqt.answer_graph.fact_scoring.fact_scoring_model import FactScoringModel
from exaqt.answer_graph.fact_scoring.dataset_fact_scoring import format_input
import exaqt.answer_graph.fact_scoring.dataset_fact_scoring as dataset
import exaqt.answer_graph.fact_scoring.dataset_temporal_fact_scoring as tempdataset
from exaqt.evaluation import answer_presence
from exaqt.answer_graph.answer_graph_construction import AnswerGraph
from exaqt.library.utils import get_config, get_logger

class FactScoringModule(AnswerGraph):
    def __init__(self, config, property):
        self.config = config
        self.property = property
        self.logger = get_logger(__name__, config)
        self.dataset = dataset.DatasetFactScoring(self.config, property)
        self.tempdataset = tempdataset.DatasetTemporalFactScoring(self.config, property)
        self.benchmark = self.config["benchmark"]
        # initialize model
        self.best_model_path = os.path.join(self.config["path_to_data"], self.benchmark, self.config["fs_model_save_path"], self.config["best_model_file"])
        self.factscore = FactScoringModel(config, self.best_model_path)
        self.model_loaded = False
        self.nerd = self.config["nerd"]

    def train(self):
        """
        Train the model on the facts of NERD entities,
        with facts retrieved by fact_retriever.
        """
        # create paths
        benchmark = self.config["benchmark"]
        input_dir = os.path.join(self.config["path_to_intermediate_results"], benchmark)

        #phase 1 process data
        train_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")
        dev_path = os.path.join(input_dir, self.nerd, f"dev-er.jsonl")
        output_path = os.path.join(input_dir, self.nerd, f"train.csv")
        train_data_loader, valid_data_loader, df_train = self.dataset.load_data(train_path, dev_path, output_path)

        #train model
        self.factscore.train(train_data_loader, valid_data_loader, df_train)

    def get_top_evidences(self, instance):
        self._load()
        with torch.no_grad():
            question = instance["Question"]
            evidences = instance["candidate_evidences"]
            # error handling for questions without evidences
            top_evidences_withscore = list()
            if not evidences:
                instance["answer_presence"] = False
            else:
                # score evidences
                relevance_scores = self.predict_scores(question, evidences)
                # sort by score
                scores, sorted_indices = torch.sort(relevance_scores, dim = 0, descending=True)
                scored_evidences = [
                    (evidences[index], relevance_scores[index]) for i, index in enumerate(sorted_indices)
                ]
                # get top evidences
                top_evidences = scored_evidences[:self.config["fs_max_evidences"]]
                for (evidence, score) in top_evidences:
                    evidence["score"] = float(score.cpu().numpy())
                    top_evidences_withscore.append(evidence)

                hit, answering_evidences = answer_presence(top_evidences_withscore, instance["answers"])
                instance["answer_presence"] = hit
        return top_evidences_withscore

    def predict_scores(self, question, evidences):
        """
        Retrieve relevance scores for the given facts/evidences.
        """
        scores = list()
        for evidence in evidences:
            question, fact = format_input(question, evidence, self.property)
            score = self.factscore.inference(question, fact)
            scores.append(score)
        scores = torch.FloatTensor(scores)
        return scores

    def _load(self):
        """Load the fine-tuned bert model."""
        # only load if not already done so
        if not self.model_loaded:
            self.factscore.load()
            self.factscore.set_eval_mode()
            self.model_loaded = True


