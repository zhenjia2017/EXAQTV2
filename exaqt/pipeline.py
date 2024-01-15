import os
import sys
import logging
import time
from pathlib import Path

from exaqt.library.utils import get_config, get_logger, get_result_logger, get_property
from exaqt.answer_graph.ftrs import FTRS
from exaqt.answer_predict.get_pretrained_embedding import get_pretrained_embedding_from_wiki2vec
from exaqt.answer_predict.get_dictionary import get_dictionary
from exaqt.answer_predict.get_relational_graph import get_subgraph
from exaqt.answer_predict.train_eva_rgcn import GCN
from exaqt.answer_predict.script_listscore import compare_pr, evaluate_result_for_category
from exaqt.evaluation import answer_presence, answer_presence_gst


class Pipeline:
    def __init__(self, config):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.property_path = os.path.join(self.config["path_to_data"], self.config["pro-info-path"])
        self.property = self._load_property()
        self.ftrs = self._load_ftrs()
        # load individual modules
        self.name = self.config["name"]
        self.benchmark = self.config["benchmark"]
        self.nerd = self.config["nerd"]
        self.fs_max_evidences = self.config["fs_max_evidences"]
        self.topg = self.config["top-gst-number"]
        self.tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        print("Loggers", loggers)

    def main_results(self):
        self.answer_graph_pipeline()
        self.answer_predict_pipeline()

    def answer_graph_pipeline(self):
        # step 0:
        # Before starting the pipeline, you need to generate the seed entities for questions.
        # Please take a look at the readme file in the package of "nerd".

        step1_start = time.time()
        self.logger.info(
            f"Step1: Start retrieve facts from nerd entities for each question in train, dev, and test dataset")
        self.ftrs.er_inference()
        self.logger.info(f"Time taken (Fact Retrieval): {time.time() - step1_start} seconds")

        self.logger.info(
            f"Step2: Start training fact scoring model, inference for each fact and keep top-f facts for each question")
        step2_start = time.time()
        self.run_ftrs()
        self.logger.info(f"Time taken (Fact Retrieval, Scoring): {time.time() - step2_start} seconds")

        self.logger.info(
            f"Step3: Start retrieve best paths between seed entities for top-f facts")
        step3_start = time.time()
        self.ftrs.path_inference()
        self.logger.info(f"Time taken (Fact Retrieval, Scoring, Path Retrieval): {time.time() - step3_start} seconds")

        self.logger.info(
            f"Step4: Start compute top-g GST and retrieve enhanced temporal facts")
        step4_start = time.time()
        self.ftrs.gst_inference()
        self.logger.info(
            f"Time taken (Fact Retrieval, Scoring, Path Retrieval, GST computation): {time.time() - step4_start} seconds")

        self.logger.info(
            f"Step5: Start training temporal fact scoring model, inference for each temporal fact and keep top-t temporal facts")
        step5_start = time.time()
        self.run_tempftrs()
        self.logger.info(
            f"Time taken (Fact Retrieval, Scoring, Path Retrieval, GST computation, Temporal fact scoring): {time.time() - step5_start} seconds")
        self.logger.info(
            f"Total Time taken for Answer Graph Pipeline: {time.time() - step1_start} seconds")

    def answer_predict_pipeline(self):
        step1_start = time.time()
        self.logger.info(f"Step1: subgraph construction: ")
        self.gcn_subgraph()
        self.logger.info(
            f"Total Time taken for Subgraph, Dictionary and Embeddings: {time.time() - step1_start} seconds")

        step2_start = time.time()
        self.logger.info(f"Step2: GCN model train: ")
        self.gcn_model_train()
        self.logger.info(f"Total Time taken for GCN model train: {time.time() - step2_start} seconds")

        step3_start = time.time()
        self.logger.info(f"Step3: Answer inference: ")
        self.gcn_model_inference()
        self.logger.info(f"Total Time taken for answer inference: {time.time() - step3_start} seconds")

        self.logger.info(
            f"Total Time taken for Answer Predict Pipeline: {time.time() - step1_start} seconds")

    def gcn_subgraph(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        fs_max_evidences = self.config["fs_max_evidences"]
        tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        # process data
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd,
                                  f"test-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_dir = os.path.join(input_dir, self.nerd, self.config["answer_predict_path"])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, f"test_subgraph.json")
        self.logger.info(
            f"Step1: Start generate subgraph for test dataset")
        get_subgraph("test", input_path, output_path, self.property, self.config)

        input_path = os.path.join(input_dir, self.nerd,
                                  f"dev-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, f"dev_subgraph.json")

        self.logger.info(
            f"Step1: Start generate subgraph for dev dataset")
        get_subgraph("dev", input_path, output_path, self.property, self.config)

        # process data

        input_path = os.path.join(input_dir, self.nerd,
                                  f"train-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, f"train_subgraph.json")

        self.logger.info(
            f"Step1: Start generate subgraph for train dataset")
        get_subgraph("train", input_path, output_path, self.property, self.config)

        self._generate_dictionary()
        self._pretrained_embeddings()

    def gcn_model_train(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        nerd = self.config["nerd"]
        data_folder = os.path.join(input_dir, nerd, 'answer_predict')
        result_path = os.path.join(data_folder, 'result')
        model_path = os.path.join(data_folder, 'model')

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        model = GCN(self.config)
        model.train()

    def gcn_model_inference(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        nerd = self.config["nerd"]
        data_folder = os.path.join(input_dir, nerd, 'answer_predict')
        result_path = os.path.join(data_folder, 'result')
        model_path = os.path.join(data_folder, 'model')

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        model = GCN(self.config)

        dev_nerd_file = os.path.join(input_dir, f"dev-nerd.json")
        dev_re_fp = result_path + '/gcn_result_dev.txt'
        dev_re_category_fp = result_path + '/gcn_category_result_dev.txt'
        # result on dev set
        dev_acc = model.dev()
        pred_kb_file = os.path.join(model_path, self.config['pred_file'])
        threshold = 0.0
        compare_pr(pred_kb_file, threshold, open(dev_re_fp, 'w', encoding='utf-8'))
        evaluate_result_for_category(dev_nerd_file, dev_re_fp, dev_re_category_fp)
        # result on test set
        test_re_fp = result_path + '/gcn_result_test.txt'
        test_re_category_fp = result_path + '/gcn_category_result_test.txt'
        test_nerd_file = os.path.join(input_dir, f"test-nerd.json")
        test_acc = model.test()
        pred_kb_file = os.path.join(model_path, self.config['pred_file'])
        compare_pr(pred_kb_file, threshold, open(test_re_fp, 'w', encoding='utf-8'))
        evaluate_result_for_category(test_nerd_file, test_re_fp, test_re_category_fp)

    def run_ftrs(self):
        self.ftrs.train()
        self.ftrs.ers_inference()

    def run_tempftrs(self):
        """Run GST on data"""
        self.ftrs.temporal_train()
        self.ftrs.temporal_ers_inference()

    def _generate_dictionary(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        dictionary_path = os.path.join(input_dir, self.nerd, 'answer_predict')

        train_path = os.path.join(dictionary_path,
                                  f"train_subgraph.json")
        dev_path = os.path.join(dictionary_path,
                                f"dev_subgraph.json")
        test_path = os.path.join(dictionary_path,
                                 f"test_subgraph.json")

        get_dictionary(dictionary_path, [train_path, dev_path, test_path])

    def _pretrained_embeddings(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        dictionary_path = os.path.join(input_dir, self.nerd, 'answer_predict')
        MODEL_FILE = os.path.join(self.config["path_to_data"], self.config['wikipedia2vec_path'])
        get_pretrained_embedding_from_wiki2vec(MODEL_FILE, dictionary_path)

    def _evaluate_gst(self, split):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run GST on data"""
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"{split}-ers-{fs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"{split}-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.ftrs.gst_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_gst_results(output_path)

    def _evaluate_gst_train(self, part):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run GST on data"""
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.ftrs.gst_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_gst_results(output_path)

    def _evaluate_retriever(self, split):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir
        input_path = os.path.join(input_dir, f"{split}-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"{split}-er.jsonl")
        self.ftrs.er_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_retrieval_results(output_path)

    def _load_ftrs(self):
        self.logger.info("Loading Fact Retrieval module")
        return FTRS(self.config, self.property)

    def _load_property(self):
        self.logger.info("Loading Property Dictionary")
        return get_property(self.property_path)


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Usage: python exaqt/pipeline.py <FUNCTION> <PATH_TO_CONFIG>")

    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    if function == "--answer-graph":
        pipeline = Pipeline(config)
        pipeline.answer_graph_pipeline()

    elif function == "--answer-predict":
        pipeline = Pipeline(config)
        pipeline.answer_predict_pipeline()

    else:
        raise Exception(f"Unknown function {function}!")
