import os
import json
from pathlib import Path
from tqdm import tqdm
from exaqt.library.utils import *
from exaqt.evaluation import answer_presence, answer_presence_gst
import time

class AnswerGraph:
    def __init__(self, config_path):
        self.config = get_config(config_path)
        self.logger = get_logger(__name__, self.config)
        self.nerd = self.config["nerd"]

    def train(self):
        """ Abstract training function that triggers training of submodules. """
        self.logger.info("Module used does not require training.")

    def er_inference(self):
        """Run ERS on data and add retrieve top-e evidences for each source combination."""
        benchmark = self.config["benchmark"]
        input_dir = os.path.join(self.config["path_to_intermediate_results"], benchmark)
        output_dir = input_dir

        # process training data
        input_path = os.path.join(input_dir, f"train-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"train-er.jsonl")
        if not os.path.exists(input_path):
            print("The nerd file needs to be created...")
            return
        self.er_inference_on_data_split(input_path, output_path)
        self.evaluate_retrieval_results(output_path)

        # process dev data
        input_path = os.path.join(input_dir, f"dev-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"dev-er.jsonl")
        if not os.path.exists(input_path):
            print("The nerd file needs to be created...")
            return
        self.er_inference_on_data_split(input_path, output_path)
        self.evaluate_retrieval_results(output_path)

        # process dev data
        input_path = os.path.join(input_dir, f"test-nerd.json")
        output_path = os.path.join(output_dir, self.nerd, f"test-er.jsonl")
        if not os.path.exists(input_path):
            print("The nerd file needs to be created...")
            return
        self.er_inference_on_data_split(input_path, output_path)
        self.evaluate_retrieval_results(output_path)

    def path_inference_on_data_split_dev_test(self, input_path, output_path):
        """
        Run Fact Retrieval on the dataset.
        """
        # score
        # create folder if not exists
        with open(input_path, "r") as fi, open(output_path, "w") as fp:
            start = time.time()
            # iterate
            datas = list()
            for line in tqdm(fi):
                instance = json.loads(line)
                datas.append(instance)

            for instance in datas:
                if datas.index(instance) < 1648: continue
                self.path_inference_on_instance(instance)
                # write instance to file
                fp.write(json.dumps(instance))
                fp.write("\n")

        # log
        self.logger.info(f"Done with seed entities pair path processing: {input_path}.")
        self.logger.info(f"Time taken (seed entities pair path): {time.time() - start} seconds")

    def path_inference_on_data_split(self, input_path, output_path):
        """
        Run Fact Retrieval on the dataset.
        """
        # score
        # create folder if not exists
        with open(input_path, "r") as fi, open(output_path, "w") as fp:
            start = time.time()
            # iterate
            datas = list()
            for line in tqdm(fi):
                instance = json.loads(line)
                datas.append(instance)

            for instance in datas:
                self.path_inference_on_instance(instance)
                # write instance to file
                fp.write(json.dumps(instance))
                fp.write("\n")

        # log
        self.logger.info(f"Done with seed entities pair path processing: {input_path}.")
        self.logger.info(f"Time taken (seed entities pair path): {time.time() - start} seconds")

    def path_connectivity_inference_on_data_split(self, input_path):
        """
        Run Fact Retrieval on the dataset.
        """
        # score
        # create folder if not exists
        count = 0
        with open(input_path, "r") as fi:
            start = time.time()
            # iterate
            datas = list()
            for line in tqdm(fi):
                instance = json.loads(line)
                datas.append(instance)

            for instance in datas:
                result = self.connectivity_check_inference_on_instance(instance)
                if result:
                    count += 1

            print (input_path)
            print (count)

        # log
        self.logger.info(f"Done with seed entities pair path processing: {input_path}.")
        self.logger.info(f"Time taken (seed entities pair path): {time.time() - start} seconds")

    def connectivity_check_inference_on_instance(self, instance):
        """Extract best path for seed pairs."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def path_inference_on_instance(self, instance):
        """Extract best path for seed pairs."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def path_inference(self):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir

        # process data
        fs_max_evidences = self.config["fs_max_evidences"]

        input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-path.jsonl")
        self.path_inference_on_data_split(input_path, output_path)

        input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path.jsonl")
        self.path_inference_on_data_split(input_path, output_path)

        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
        self.path_inference_on_data_split(input_path, output_path)

    def gst_inference(self):
        """Run GST on data and add retrieve top-e evidences for each source combination."""
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir

        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.gst_inference_on_data_split(input_path, output_path)
        self.evaluate_gst_results(output_path)

        input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.gst_inference_on_data_split(input_path, output_path)
        self.evaluate_gst_results(output_path)

        input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-gst-{topg}.jsonl")

        self.gst_inference_on_data_split(input_path, output_path)
        self.evaluate_gst_results(output_path)

    def gst_inference_on_data_split(self, input_path, output_path):
        start = time.time()
        # process data
        with open(input_path, 'r') as fp, open(output_path, "w") as fout:
            data = list()
            for line in tqdm(fp):
                instance = json.loads(line)
                self.gst_inference_on_instance(instance)
                if instance["complete_gst_entity"]:
                    gst_hit = answer_presence_gst(instance["complete_gst_entity"], instance["answers"])
                    instance["gst_answer_presence"] = gst_hit
                    temporal_enhance_entities = list()
                    for evidence in instance["temporal_evidences"]:
                        temporal_enhance_entities += evidence["wikidata_entities"]

                    temporal_enhance_hit = answer_presence_gst(temporal_enhance_entities, instance["answers"])
                    instance["temporal_enhance_answer_presence"] = gst_hit or temporal_enhance_hit
                else:
                    instance["gst_answer_presence"] = False
                    instance["temporal_enhance_answer_presence"] = False
                # write instance to file
                fout.write(json.dumps(instance))
                fout.write("\n")
                data.append(instance)

            gst_answer_presence_list = [instance["gst_answer_presence"] for instance in data]
            print(f"Time taken: {time.time() - start}")
            print(f"Num Questions: {len(gst_answer_presence_list)}")
            print(f"GST Answer presence: {sum(gst_answer_presence_list) / len(gst_answer_presence_list)}")

            temporal_enhannce_answer_presence_list = [instance["temporal_enhance_answer_presence"] for instance in data]
            print(f"Time taken: {time.time() - start}")
            print(f"Num Questions: {len(temporal_enhannce_answer_presence_list)}")
            print(f"Temporal Enhance Answer presence: {sum(temporal_enhannce_answer_presence_list) / len(temporal_enhannce_answer_presence_list)}")
        # log
        self.logger.info(f"GST results: {output_path}.")
        self.logger.info(f"Done with processing: {input_path}.")

    def _format_answers(self, instance):
        # TBD:  JZ need to be updated if answer format will be changed for Temp.Ans questions
        """
        Reformat answers in the timequestions dataset.
        """
        _answers = list()
        for answer in instance["Answer"]:
            if answer["AnswerType"] == "Entity":
                answer = {"id": answer["WikidataQid"], "label": answer["WikidataLabel"]}
            elif answer["AnswerType"] == "Value":
                answer = {
                    "id": answer["AnswerArgument"],
                    "label": convert_timestamp_to_date(
                        answer["AnswerArgument"]) if is_timestamp(answer["AnswerArgument"]) else answer[
                        "AnswerArgument"]
                }
            elif answer["AnswerType"] == "Timestamp":
                answer = {
                    "id": answer["AnswerArgument"],
                    "label": convert_timestamp_to_date(
                        answer["AnswerArgument"]) if is_timestamp(answer["AnswerArgument"]) else answer[
                        "AnswerArgument"]
                }
            # elif answer["AnswerType"] == "Timespan":

            else:
                print(answer)
                raise Exception
            _answers.append(answer)
        return _answers

    def er_inference_on_data_split(self, input_path, output_path):
        """
        Run Fact Retrieval on the dataset.
        """
        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)
        self.logger.info(f"Input data loaded from: {input_path}.")

        # score
        # create folder if not exists
        output_dir = os.path.dirname(output_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # process data
        with open(output_path, "w") as fp:
            for instance in tqdm(data):
                evidences = self.er_inference_on_instance(instance)
                instance["answers"] = self._format_answers(instance)
                # answer presence
                hit, answering_evidences = answer_presence(evidences, instance["answers"])
                instance["answer_presence"] = hit

                # write instance to file
                fp.write(json.dumps(instance))
                fp.write("\n")

        # log
        self.logger.info(f"Evaluating retrieval results: {output_path}.")
        self.evaluate_retrieval_results(output_path)
        self.logger.info(f"Done with processing: {input_path}.")

    def er_inference_on_data(self, input_data):
        """Run model on data and add predictions."""
        # model inference on given data
        for instance in tqdm(input_data):
            self.er_inference_on_instance(instance)
        return input_data

    def gst_inference_on_data(self, input_data):
        """Run model on data and add predictions."""
        # model inference on given data
        for instance in tqdm(input_data):
            self.gst_inference_on_instance(instance)
        return input_data

    def er_inference_on_instance(self, instance):
        """Retrieve candidate facts."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def gst_inference_on_instance(self, instance):
        """Run gst for an instance."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def tempers_inference_on_instance(self, instance):
        """Retrieve candidate temporal facts."""
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def ers_inference_on_instance(self, instance):
        """Run model on data and add predictions."""
        # inference: add predictions to data
        """ Abstract evidence scoring function that triggers es_inference_on_instance of submodules. """
        raise Exception("This is an abstract function which should be overwritten in a derived class!")


    def ers_inference(self):
        self.ers_inference_split("test")
        self.ers_inference_split("train")
        self.ers_inference_split("dev")

    def ers_inference_split(self, split):
        fs_max_evidences = self.config["fs_max_evidences"]
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir

        # process data
        input_path = os.path.join(input_dir, self.nerd, f"{split}-er.jsonl")
        output_path = os.path.join(output_dir, self.nerd, f"{split}-ers-{fs_max_evidences}.jsonl")

        self.ers_inference_on_data_split(input_path, output_path)
        self.evaluate_retrieval_results(output_path)

    def temporal_ers_inference(self):
        self.temporal_ers_inference_split("test")
        self.temporal_ers_inference_split("dev")
        self.temporal_ers_inference_split("train")

    def temporal_ers_inference_split(self, split):
        input_dir = os.path.join(self.config["path_to_intermediate_results"], self.config["benchmark"])
        output_dir = input_dir

        fs_max_evidences = self.config["fs_max_evidences"]
        topg = self.config["top-gst-number"]
        tempfs_max_evidences = self.config["temporal_fs_max_evidences"]
        # process data
        input_path = os.path.join(input_dir, self.nerd, f"{split}-ers-{fs_max_evidences}-gst-{topg}.jsonl")
        output_path = os.path.join(output_dir, self.nerd,
                                   f"{split}-ers-{fs_max_evidences}-gst-{topg}-{tempfs_max_evidences}.jsonl")
        self.tempers_inference_on_data_split(input_path, output_path)

    def tempers_inference_on_data_split(self, input_path, output_path):
        # score
        # create folder if not exists
        """
                Retrieve the best evidences among the
                given list of evidences for the current question.
                """
        with open(input_path, "r") as fi, open(output_path, "w") as fo:

            # iterate
            data = list()
            for line in tqdm(fi):
            # line in fi:
                instance = json.loads(line)
                start = time.time()
                temp_top_evidences = self.tempers_inference_on_instance(instance)
                self.logger.info(f"Time taken (inference): {time.time() - start} seconds")
                data.append(instance)

                candidate_entities = instance["complete_gst_entity"]
                hit = answer_presence_gst(candidate_entities, instance["answers"])
                temp_hit = instance["temporal_answer_presence"]
                instance["enhance_temporal_answer_presence"] = hit or temp_hit
                fo.write(json.dumps(instance))
                fo.write("\n")

            answer_presence_list = [instance["enhance_temporal_answer_presence"] for instance in data]
            print(f"Answer presence ({len(answer_presence_list)}): {sum(answer_presence_list) / len(answer_presence_list)}")
            print(f"Time taken: {time.time() - start}")
            print(f"Num Questions: {len(answer_presence_list)}")
            print(f"Temporal Answer presence: {sum(answer_presence_list) / len(answer_presence_list)}")

    def evaluate_tempers_inference_results(self, results_path):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        # score
        answer_presences = list()
        category_to_temporal = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_enhanced = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}

        count = 0
        # process data
        # with open(results_path, "r") as fp:
        #     data = json.load(fp)
        #     for instance in tqdm(data):
        with open(results_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                category_slot = [cat.lower() for cat in instance["Temporal question type"]]
                temp_hit = instance["temporal_answer_presence"]
                enhanced_temp_hit = instance["enhance_temporal_answer_presence"]
                category_to_temporal["all"] += [temp_hit]
                category_to_enhanced["all"] += [enhanced_temp_hit]
                for category in category_to_temporal.keys():
                    if category in category_slot:
                        category_to_temporal[category] += [temp_hit]

                for category in category_to_enhanced.keys():
                    if category in category_slot:
                        category_to_enhanced[category] += [enhanced_temp_hit]

        # print results
        res_path = results_path.replace(".jsonl", ".res")
        print (len(category_to_temporal["all"]))
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_temporal.items() if len(num) != 0
            }
            fp.write(f"\nCategory Answer presence per source for temporal facts: {category_answer_presence_per_src}")
            fp.write(f"\nevaluation result:\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_enhanced.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source for enhanced temporal facts: {category_answer_presence_per_src}")


    def ers_inference_on_data_split(self, input_path, output_path):
        # score
        """
        Retrieve the best evidences among the
        given list of evidences for the current question.
        """
        with open(input_path, "r") as fi, open(output_path, "w") as fo:
            start = time.time()
            # iterate
            data = list()
            for line in tqdm(fi):
                instance = json.loads(line)
                top_evidences = self.ers_inference_on_instance(instance)
                # store data
                fo.write(json.dumps(instance))
                fo.write("\n")

                data.append(instance)
                answer_presence_list = [instance["answer_presence"] for instance in data]
                print(f"Answer presence ({len(answer_presence_list)}): {sum(answer_presence_list) / len(answer_presence_list)}")

            answer_presence_list = [instance["answer_presence"] for instance in data]
            print(f"Time taken: {time.time() - start}")
            print(f"Num Questions: {len(answer_presence_list)}")
            print(f"Answer presence: {sum(answer_presence_list) / len(answer_presence_list)}")

    def evaluate_gst_results(self, results_path):
        """
                Evaluate the results of the gst phase, for
                each source, and for each category.
                """
        # score
        answer_presences = list()
        category_to_ans_pres = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_ans_pres_tempenhance = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}

        with open(results_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                category_slot = [cat.lower() for cat in instance["Temporal question type"]]
                candidate_entities = instance["complete_gst_entity"]
                hit = answer_presence_gst(candidate_entities, instance["answers"])
                category_to_ans_pres["all"] += [hit]
                category_to_ans_pres_tempenhance["all"] += [instance["temporal_enhance_answer_presence"]]
                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_ans_pres[category] += [hit]
                        category_to_ans_pres_tempenhance[category] += [instance["temporal_enhance_answer_presence"]]

        # print results
        res_path = results_path.replace(".jsonl", "-gst.res")
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

            fp.write(f"evaluation result after enhancing temporal facts:\n")
            category_answer_presence_per_src_tempenhance = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres_tempenhance.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source after enhancing temporal facts: {category_answer_presence_per_src_tempenhance}")

    def evaluate_retrieval_results(self, results_path):
        """
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
        # score
        answer_presences = list()
        category_to_ans_pres = {"explicit": [], "implicit": [], "temp.ans": [], "ordinal": [], "all": []}
        category_to_evi_num = {"explicit": [], "implicit": [], "temp.ans": [], "ordinal":[], "all": []}
        #category_to_ans_pres = {"implicit": [], "all": []}
        #category_to_evi_num = {"implicit": [], "all": []}
        # process data
        # with open(results_path, "r") as fp:
        #     data = json.load(fp)
        #     for instance in tqdm(data):
        with open(results_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                category_slot = [cat.lower() for cat in instance["Temporal question type"]]
                candidate_evidences = instance["candidate_evidences"]
                hit, answering_evidences = answer_presence(candidate_evidences, instance["answers"])

                category_to_ans_pres["all"] += [hit]
                category_to_evi_num["all"] += [len(candidate_evidences)]
                for category in category_to_ans_pres.keys():
                    if category in category_slot:
                        category_to_evi_num[category] += [len(candidate_evidences)]
                        category_to_ans_pres[category] += [hit]

                answer_presences += [hit]

        # print results
        res_path = results_path.replace(".jsonl", ".res")
        with open(res_path, "w") as fp:
            fp.write(f"evaluation result:\n")
            category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
            }
            fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

            for category in category_to_evi_num:
                print(category)
                print(len(category_to_evi_num[category]))

            for category in category_to_evi_num:
                fp.write("\n")
                fp.write(f"category: {category}\n")
                if len(category_to_evi_num[category]) == 0: continue
                fp.write(f"Avg. evidence number: {sum(category_to_evi_num[category]) / len(category_to_evi_num[category])}\n")
                sorted_category_num = category_to_evi_num[category]
                sorted_category_num.sort()
                fp.write(f"Max. evidence number: {sorted_category_num[-1]}\n")
                fp.write(f"Min. evidence number: {sorted_category_num[0]}\n")


