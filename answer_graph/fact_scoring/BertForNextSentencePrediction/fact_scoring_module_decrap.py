import os.path
import time
import torch

from exaqt.answer_graph.fact_scoring.fact_scoring_model import FactScoringModel
from exaqt.answer_graph.fact_scoring.dataset_fact_scoring import format_input
from exaqt.evaluation import answer_presence
from exaqt.answer_graph.answer_graph_construction import AnswerGraph
from exaqt.library.utils import get_config, get_logger

class FactScoringModule(AnswerGraph):
    """
    Class to ensure that evidence relevance can be retrieved easily.
    Interface between core-code and EvidenceRelevancePrediction.
    """
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        # initialize model
        self.factscore = FactScoringModel(config)
        self.model_loaded = False
        self.nerd = self.config["nerd"]

    def train(self):
        """
        Train the model on silver evidence relevances,
        with evidences retrieved by evidence_retrieval.
        """
        # create paths
        input_dir = self.config["path_to_intermediate_results"]

        # process data
        train_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")
        dev_path = os.path.join(input_dir, self.nerd, f"dev-er.jsonl")

        # train model
        self.factscore.train(train_path, dev_path)

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
                scored_evidences = self.predict_scores(question, evidences)

                # gather scores for relevance
                relevance_scores = scored_evidences[:, 1]

                # sort by score
                scores, sorted_indices = torch.sort(relevance_scores, descending=True)

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
        Retrieve relevance scores for the given evidences
        for the current AR. The process is done in batches
        for efficiency.
        """
        test_batch_size = self.config["fs_test_batch_size"]

        # initialize
        relevances = None
        number_of_batches = (len(evidences) // test_batch_size) + 1

        # batch-processing
        start_time = time.time()
        for i in range(number_of_batches):
            # define start and end indexes for batch
            start_index = i * test_batch_size
            end_index = min(len(evidences), (i + 1) * test_batch_size)

            # fetch inputs
            batch_evidences = evidences[start_index:end_index]
            if not batch_evidences:
                break
            batch_inputs = [format_input(question, evidence) for evidence in batch_evidences]

            # concat new relevance scores for batch
            if relevances is None:
                relevances = self.factscore.inference(batch_inputs)
            else:
                relevances = torch.cat((relevances, self.factscore.inference(batch_inputs)))
        self.logger.debug(f"Finished! Total batches: {number_of_batches}, Num. evidences: {len(evidences)}, time consumed: {time.time()-start_time}")
        relevances = torch.sigmoid(relevances)
        return relevances

    def _load(self):
        """Load the fine-tuned bert model."""
        # only load if not already done so
        if not self.model_loaded:
            self.factscore.load()
            self.factscore.set_eval_mode()
            self.model_loaded = True

#######################################################################################################################
#######################################################################################################################
import sys
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("python exaqt/answer_graph/fact_scoring/fact_scoring_module.py <PATH_TO_CONFIG>")
    # load config
    config_path = sys.argv[1]
    config = get_config(config_path)
    fs = FactScoringModule(config)
    input_dir = config["path_to_intermediate_results"]
    output_dir = config["path_to_intermediate_results"]
    nerd = config["nerd"]
    fs_max_evidences = config["fs_max_evidences"]
    # process data
    input_path = os.path.join(input_dir, nerd, f"test-er.jsonl")
    output_path = os.path.join(output_dir, nerd, f"test-ers-{fs_max_evidences}.jsonl")

    fs.ers_inference_on_data_split(input_path, output_path)
    fs.evaluate_retrieval_results(output_path)
        #
        # input_path = os.path.join(input_dir, self.nerd, f"dev-er.jsonl")
        # output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")
        #
        # self.ers_inference_on_data_split(input_path, output_path)
        # self.evaluate_retrieval_results(output_path)
        #
        # input_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")
        # output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}.jsonl")
        #
        # self.ers_inference_on_data_split(input_path, output_path)
        # self.evaluate_retrieval_results(output_path)

