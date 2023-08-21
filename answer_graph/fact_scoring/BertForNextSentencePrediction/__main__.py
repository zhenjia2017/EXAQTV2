import os
import sys
import json

from tqdm import tqdm

from fact_scoring_model import FactScoringModel
from evidence_relevance_retriever import EvidenceRelevanceRetriever
from convinse.library.utils import get_config

def train(config):
    """
    Train the model on silver evidence relevances,
    with evidences retrieved by evidence_retrieval.
    """
    # create paths
    input_dir = config["path_to_intermediate_in"]
    train_path = os.path.join(input_dir, "convinse_er_train.jsonl")
    dev_path = os.path.join(input_dir, "convinse_er_dev.jsonl")

    # train model
    ers = FactScoringModel(config)
    ers.train(train_path, dev_path)

def inference(config):
    """Run the model on the dataset to predict answering evidences for each AR."""
    # create paths
    input_dir = config["path_to_intermediate_in"]

    # create relevance retriever
    err = EvidenceRelevanceRetriever(config)

    # inference
    input_path = os.path.join(input_dir, "convinse_er_test.jsonl")
    output_path = os.path.join(input_dir, "convinse_er_test_scored.jsonl")
    err.predict_top_evidences(input_path, output_path)


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    args = sys.argv[1:] 
    config_path = args[1] if len(args) > 1 else "CONVINSE/config/timequestions/ehtqa.yml"
    config = get_config(config_path)

    # train: train evidence scoring module
    if args[0] == "--train":
        train(config)

    # inference: score evidence-question pairs and evaluate top evidences with QA metrics
    elif args[0] == "--inference":
        inference(config)

    else:
        raise Exception("python neural_es <TASK> <PATH_TO_CONFIG>")
