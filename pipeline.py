import os
import sys
import logging
import time
import json
from tqdm import tqdm

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
        self.property_path = self.config["pro-info-path"]
        self.property = self._load_property()
        self.ftrs = self._load_ftrs()
        # load individual modules
        self.name = self.config["name"]
        self.nerd = self.config["nerd"]
        self.fs_max_evidences = self.config["fs_max_evidences"]
        self.topg = self.config["top-gst-number"]
        self.tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        print("Loggers", loggers)

    def answer_graph_construction_pipeline(self):
        start = time.time()
        self.logger.info(
            f"Step1: Start retrieve facts from nerd entities for each question in train, dev, and test dataset")
        self.ftrs.er_inference()
        self.logger.info(f"Time taken (Fact Retrieval): {time.time() - start} seconds")

        self.logger.info(
            f"Step2: Start training fact scoring model, inference for each fact and keep top-f facts for each question")
        self.run_ftrs()
        self.logger.info(f"Time taken (Fact Retrieval, Scoring): {time.time() - start} seconds")

        self.logger.info(
            f"Step3: Start retrieve best paths between seed entities for top-f facts")
        self.ftrs.path_inference()
        self.logger.info(f"Time taken (Fact Retrieval, Scoring, Path Retrieval): {time.time() - start} seconds")

        self.logger.info(
            f"Step4: Start compute top-g GST and retrieve enhanced temporal facts")
        self.ftrs.gst_inference()
        self.logger.info(
            f"Time taken (Fact Retrieval, Scoring, Path Retrieval, GST computation): {time.time() - start} seconds")

        self.logger.info(
            f"Step5: Start training temporal fact scoring model, inference for each temporal fact and keep top-t temporal facts")
        self.run_tempftrs()
        self.logger.info(
            f"Time taken (Fact Retrieval, Scoring, Path Retrieval, GST computation, Temporal fact scoring): {time.time() - start} seconds")

    def gcn_subgraph(self):
        input_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        # process data
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd,
                                  f"test-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_path = os.path.join(input_dir, self.nerd, 'answer_predict', f"test_subgraph.json")

        self.logger.info(
            f"Step1: Start generate subgraph for test dataset")
        #get_subgraph("test", input_path, output_path, self.property, self.config)

        input_path = os.path.join(input_dir, self.nerd,
                                  f"dev-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_path = os.path.join(input_dir, self.nerd, 'answer_predict', f"dev_subgraph.json")

        self.logger.info(
            f"Step1: Start generate subgraph for dev dataset")
        #get_subgraph("dev", input_path, output_path, self.property, self.config)

        # process data
        input_path = os.path.join(input_dir, self.nerd,
                                  f"train-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        output_path = os.path.join(input_dir, self.nerd, 'answer_predict',
                                   f"train_subgraph.json")
        self.logger.info(
            f"Step1: Start generate subgraph for train dataset")
        #get_subgraph("train", input_path, output_path, self.property, self.config)

        self._generate_dictionary()
        self._pretrained_embeddings()

    def gcn_model_train(self):
        input_dir = self.config["path_to_intermediate_results"]
        nerd = self.config["nerd"]
        data_folder = os.path.join(input_dir, nerd, 'answer_predict')
        result_path = os.path.join(data_folder, 'result')
        model_path = os.path.join(data_folder, 'model')

        os.makedirs(result_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        model = GCN(self.config)
        model.train()

    def gcn_model_inference(self):
        input_dir = self.config["path_to_intermediate_results"]
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
        #evaluate_result_for_category(dev_nerd_file, dev_re_fp, dev_re_category_fp)
        # result on test set
        test_re_fp = result_path + '/gcn_result_test.txt'
        test_re_category_fp = result_path + '/gcn_category_result_test.txt'
        test_nerd_file = os.path.join(input_dir, f"test-nerd.json")
        test_acc = model.test()
        pred_kb_file = os.path.join(model_path, self.config['pred_file'])
        compare_pr(pred_kb_file, threshold, open(test_re_fp, 'w', encoding='utf-8'))
        #evaluate_result_for_category(test_nerd_file, test_re_fp, test_re_category_fp)

    def run_ftrs(self):
        self.ftrs.train()

    # self.ftrs.ers_inference()

    def run_ftrs_inference_test(self):
        self.ftrs.ers_inference_test()

    def run_ftrs_inference_dev(self):
        self.ftrs.ers_inference_dev()

    def run_ftrs_inference_train(self):
        self.ftrs.ers_inference_train()

    def run_tempftrs(self):
        """Run GST on data"""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]

        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-gst-{topg}-gst.res")
        print(output_path)
        while not os.path.exists(output_path):
            print("Waiting for the file to be created...")
            time.sleep(30)  # You can adjust the sleep interval as needed

        print("File has been created!")
        self.ftrs.temporal_train()
        self.ftrs.temporal_ers_inference()

    def run_tempers_inference_test(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}-gst-{topg}-reverse.jsonl")
        output_path = os.path.join(output_dir, self.nerd,
                                   f"test-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}-reverse.jsonl")

        self.ftrs.tempers_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_tempers_inference_results(output_path)

    def run_tempers_inference_train(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{self.fs_max_evidences}-gst-{self.topg}.jsonl")
        output_path = os.path.join(output_dir, self.nerd,
                                   f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}.jsonl")

        self.ftrs.tempers_inference_on_data_split(input_path, output_path)

    def run_tempers_inference_train_part(self, part):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{self.fs_max_evidences}-gst-{self.topg}.jsonl")

        data = []
        with open(input_path, "r") as fi:
            for line in tqdm(fi):
                instance = json.loads(line)
                data.append(instance)
        self.logger.info(f"Input data loaded from: {input_path}.")

        if part == "part1":
            part_instances = data[0:1000]
        elif part == "part2":
            part_instances = data[1000:2000]
        elif part == "part3":
            part_instances = data[2000:3000]
        elif part == "part4":
            part_instances = data[3000:]

        self.logger.info(f"Split Input data into three parts: {len(part_instances)}.")

        input_path_part = os.path.join(input_dir, self.nerd,
                                       f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{part}.jsonl")

        with open(input_path_part, "w") as fout:
            for instance in part_instances:
                # write instance to file
                fout.write(json.dumps(instance))
                fout.write("\n")

        output_path = os.path.join(output_dir, self.nerd,
                                   f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-{part}.jsonl")

        self.ftrs.tempers_inference_on_data_split(input_path_part, output_path)
        self.ftrs.evaluate_tempers_inference_results(output_path)

    def merge_parts_test(self, dev=False):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        if dev:
            input_path_part1 = os.path.join(input_dir, self.nerd,
                                            f"dev-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part1.jsonl")
            input_path_part2 = os.path.join(input_dir, self.nerd,
                                            f"dev-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-reverse.jsonl")
            output_path = os.path.join(input_dir, self.nerd,
                                   f"dev-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}.jsonl")

        else:
            input_path_part1 = os.path.join(input_dir, self.nerd,
                                        f"test-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part1.jsonl")
            input_path_part2 = os.path.join(input_dir, self.nerd,
                                        f"test-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-reverse.jsonl")
            output_path = os.path.join(input_dir, self.nerd,
                                   f"test-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}.jsonl")

        data = []
        for input_path in [input_path_part1, input_path_part2]:
            with open(input_path, "r") as fi:
                for line in tqdm(fi):
                    instance = json.loads(line)
                    data.append(instance)

        sorted_list = sorted(data, key=lambda x: x['Id'])

        with open(output_path, "w") as fout:
            for instance in sorted_list:
                fout.write(json.dumps(instance))
                fout.write("\n")

        self.ftrs.evaluate_tempers_inference_results(output_path)

    def merge_parts(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        input_path_part1 = os.path.join(input_dir, self.nerd,
                                        f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part1.jsonl")
        input_path_part2 = os.path.join(input_dir, self.nerd,
                                        f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part2.jsonl")
        input_path_part3 = os.path.join(input_dir, self.nerd,
                                        f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part3.jsonl")
        input_path_part4 = os.path.join(input_dir, self.nerd,
                                        f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-part4.jsonl")

        # input_path_part2 = os.path.join(input_dir, self.nerd,
        #                                 f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-part2.jsonl")
        # input_path_part3 = os.path.join(input_dir, self.nerd,
        #                                 f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-part3.jsonl")
        # input_path_part4 = os.path.join(input_dir, self.nerd,
        #                                 f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-part4.jsonl")
        data = []
        for input_path in [input_path_part1, input_path_part2, input_path_part3, input_path_part4]:
            with open(input_path, "r") as fi:
                for line in tqdm(fi):
                    instance = json.loads(line)
                    data.append(instance)

        sorted_list = sorted(data, key=lambda x: x['Id'])
        output_path = os.path.join(input_dir, self.nerd,
                                        f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}.jsonl")

        # output_path = os.path.join(output_dir, self.nerd,
        #                            f"train-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}.jsonl")

        with open(output_path, "w") as fout:
            for instance in sorted_list:
                fout.write(json.dumps(instance))
                fout.write("\n")

        self.ftrs.evaluate_tempers_inference_results(output_path)

    def run_tempers_inference_dev(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        # process data
        input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{self.fs_max_evidences}-gst-{self.topg}-reverse.jsonl")
        output_path = os.path.join(output_dir, self.nerd,
                                   f"dev-ers-{self.fs_max_evidences}-gst-{self.topg}-{self.tempfs_max_evidences}-reverse.jsonl")

        self.ftrs.tempers_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_tempers_inference_results(output_path)

    def run_ers_inference_test(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"test-er.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}.jsonl")

        self.ftrs.ers_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_retrieval_results(output_path)

    def run_ers_inference_dev(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"dev-er.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")

        self.ftrs.ers_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_retrieval_results(output_path)

    def run_ers_inference_train_part(self, part):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")

        data = []
        with open(input_path, "r") as fi:
            for line in tqdm(fi):
                instance = json.loads(line)
                data.append(instance)
        self.logger.info(f"Input data loaded from: {input_path}.")

        if part == "part1":
            part_instances = data[2890:2900]
        elif part == "part2":
            part_instances = data[2900:2950]
        elif part == "part3":
            part_instances = data[2950:3000]
        elif part == "part4":
            part_instances = data[3000:]

        self.logger.info(f"Split Input data into three parts: {len(part_instances)}.")

        input_path_part = os.path.join(input_dir, self.nerd, f"train-er-{part}.jsonl")

        with open(input_path_part, "w") as fout:
            for instance in part_instances:
                # write instance to file
                fout.write(json.dumps(instance))
                fout.write("\n")

        output_path_part = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-{part}.jsonl")

        self.ftrs.ers_inference_on_data_split(input_path_part, output_path_part)
        self.ftrs.evaluate_retrieval_results(output_path_part)

    def run_ers_inference_train(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}.jsonl")

        self.ftrs.ers_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_retrieval_results(output_path)

    # def run_tempftrs(self):
    # 	self.ftrs.temporal_train()
    # 	self.ftrs.temporal_ers_inference()

    def _generate_dictionary(self):
        input_dir = self.config["path_to_intermediate_results"]
        train_path = os.path.join(input_dir, self.nerd, 'answer_predict',
                                  f"train_subgraph.json")
        dev_path = os.path.join(input_dir, self.nerd, 'answer_predict',
                                f"dev_subgraph.json")
        test_path = os.path.join(input_dir, self.nerd, 'answer_predict',
                                 f"test_subgraph.json")

        dictionary_path = os.path.join(input_dir, self.nerd, 'answer_predict')
        get_dictionary(dictionary_path, [train_path, dev_path, test_path])

    def _pretrained_embeddings(self):
        input_dir = self.config["path_to_intermediate_results"]
        dictionary_path = os.path.join(input_dir, self.nerd, 'answer_predict')
        MODEL_FILE = self.config['wikipedia2vec_path']
        get_pretrained_embedding_from_wiki2vec(MODEL_FILE, dictionary_path)

    def _evaluate_seed_path_test(self):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run ERS on data and add retrieve top-e evidences for each source combination."""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}.jsonl")
        # process data
        output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-path-reverse.jsonl")
        self.ftrs.path_inference_on_data_split(input_path, output_path)

    def _evaluate_seed_path_dev(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path-reverse.jsonl")
        self.ftrs.path_inference_on_data_split(input_path, output_path)


    def _evaluate_seed_path_train_part_split(self, part):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-part0.jsonl")
        data = []
        with open(input_path, "r") as fi:
            for line in tqdm(fi):
                instance = json.loads(line)
                data.append(instance)
        self.logger.info(f"Input data loaded from: {input_path}.")

        if part == "part01":
            part_instances = data[534:1000]
        elif part == "part02":
            part_instances = data[1000:1500]
        elif part == "part03":
            part_instances = data[1500:2000]
        elif part == "part04":
            part_instances = data[2000:2500]
        elif part == "part05":
            part_instances = data[2500:2613]
        elif part == "part07":
            part_instances = data[2770:2891]

        input_path_part = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-{part}.jsonl")

        with open(input_path_part, "w") as fout:
            for instance in part_instances:
                # write instance to file
                fout.write(json.dumps(instance))
                fout.write("\n")

        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-path-{part}.jsonl")
        self.ftrs.path_inference_on_data_split(input_path_part, output_path)

    def _evaluate_seed_path_train(self, part):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-{part}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-path-{part}.jsonl")
        self.ftrs.path_inference_on_data_split(input_path, output_path)

    def _evaluate_gst_test(self):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run GST on data"""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}-path-reverse.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-gst-{topg}-reverse.jsonl")

        self.ftrs.gst_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_gst_results(output_path)

    def _evaluate_gst_dev(self):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run GST on data"""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path-reverse.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-gst-{topg}-reverse.jsonl")

        self.ftrs.gst_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_gst_results(output_path)

    def _evaluate_gst_train(self):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run GST on data"""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.ftrs.gst_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_gst_results(output_path)

    def _evaluate_retriever_test(self):
        """
        Run the pipeline using gold answers for the dataset.
        """
        # define output path
        """Run ERS on data and add retrieve top-e evidences for each source combination."""
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        # process data
        input_path = os.path.join(input_dir, f"test-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"test-er.jsonl")
        # self.ftrs.er_inference_on_data_split(input_path, output_path)

        self.ftrs.evaluate_retrieval_results(output_path)

    def _evaluate_retriever_dev(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        input_path = os.path.join(input_dir, f"dev-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"dev-er.jsonl")
        # self.ftrs.er_inference_on_data_split(input_path, output_path)
        self.ftrs.evaluate_retrieval_results(output_path)

    def _evaluate_retriever_train(self):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]
        input_path = os.path.join(input_dir, f"train-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"train-er.jsonl")
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

    # inference using predicted answers
    if function == "--path-test":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_test()

    elif function == "--path-dev":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_dev()

    elif function == "--path-train":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train()

    elif function == "--path-train-01":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part01")

    elif function == "--path-train-02":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part02")

    elif function == "--path-train-03":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part03")

    elif function == "--path-train-04":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part04")

    elif function == "--path-train-05":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part05")

    elif function == "--path-train-07":
        pipeline = Pipeline(config)
        pipeline._evaluate_seed_path_train_part_split("part07")

    # inference using predicted answers
    elif function == "--retrieve-test":
        pipeline = Pipeline(config)
        pipeline._evaluate_retriever_test()

    elif function == "--retrieve-dev":
        pipeline = Pipeline(config)
        pipeline._evaluate_retriever_dev()

    elif function == "--retrieve-train":
        pipeline = Pipeline(config)
        pipeline._evaluate_retriever_train()

    elif function == "--run_ftrs":
        pipeline = Pipeline(config)
        pipeline.run_ftrs()

    elif function == "--run_tempftrs":
        pipeline = Pipeline(config)
        pipeline.run_tempftrs()

    elif function == "--run_tempftrs_test":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_test()

    elif function == "--run_tempftrs_dev":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_dev()

    elif function == "--run_tempftrs_train":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_train()

    elif function == "--run_tempftrs_train-part1":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_train_part("part1")

    elif function == "--run_tempftrs_train-part2":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_train_part("part2")

    elif function == "--run_tempftrs_train-part3":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_train_part("part3")

    elif function == "--run_tempftrs_train-part4":
        pipeline = Pipeline(config)
        pipeline.run_tempers_inference_train_part("part4")

    elif function == "--gst-test":
        pipeline = Pipeline(config)
        pipeline._evaluate_gst_test()

    elif function == "--gst-dev":
        pipeline = Pipeline(config)
        pipeline._evaluate_gst_dev()

    elif function == "--gst-train":
        pipeline = Pipeline(config)
        pipeline._evaluate_gst_train()

    elif function == "--fts-test":
        pipeline = Pipeline(config)
        pipeline.run_ers_inference_test()

    elif function == "--fts-dev":
        pipeline = Pipeline(config)
        pipeline.run_ers_inference_dev()

    elif function == "--fts-train":
        pipeline = Pipeline(config)
        pipeline.run_ers_inference_train()

    elif function == "--gcn-subgraph":
        pipeline = Pipeline(config)
        pipeline.gcn_subgraph()

    elif function == "--gcn-model-train":
        pipeline = Pipeline(config)
        pipeline.gcn_model_train()

    elif function == "--gcn-model-inference":
        pipeline = Pipeline(config)
        pipeline.gcn_model_inference()

    elif function == "--gcn-pipeline":
        pipeline = Pipeline(config)
        pipeline.gcn_subgraph()
        pipeline.gcn_model_train()
        pipeline.gcn_model_inference()

    elif function == "--merge-part":
        pipeline = Pipeline(config)
        pipeline.merge_parts()

    elif function == "--merge-test":
        pipeline = Pipeline(config)
        pipeline.merge_parts_test()

    elif function == "--merge-dev":
        pipeline = Pipeline(config)
        pipeline.merge_parts_test(dev=True)

    else:
        raise Exception(f"Unknown function {function}!")
