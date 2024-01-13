import truecase
import json
import random
from tqdm import tqdm
import csv
from exaqt.evaluation import answer_presence, evidence_has_answer
from exaqt.library.utils import get_config, get_logger, get_result_logger, get_property
from exaqt.answer_graph.fact_retriever.fact_er import FactRetriever
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
# set seed for reproducability
random.seed(7)

class CreateDatasetTemporalFactScoring:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.property_path = os.path.join(self.config["path_to_data"], self.config["pro-info-path"])
        self.property = self._load_property()
        self.nerd = self.config["nerd"]
        self.fact_retriever = FactRetriever(config, self.property)
        # initialize clocq for KB-facts
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()
        self.max_pos_examples = self.config["temporal_fs_max_pos_evidences"]
        self.max_neg_examples = self.config["temporal_fs_max_neg_evidences"]

    def _load_property(self):
        self.logger.info("Loading Property Dictionary")
        return get_property(self.property_path)

    def create_train_data(self, input_path, output_path):
        train_set_instances = []
        train_list = []
        count = 0
        # open data
        print("Transform data")
        with open(input_path, 'r') as fp:
            for line in tqdm(fp):
                instance = json.loads(line)
                train_set_instances.append(instance)

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
                positive_train = random.sample(positive, self.max_pos_examples)
                if len(negative) < 5:
                    negative_train = random.sample(negative, len(negative))
                else:
                    negative_train = random.sample(negative, self.max_neg_examples)
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

#######################################################################################################################
import sys
import os
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("python exaqt/answer_graph/fact_scoring/fact_scoring_module.py <PATH_TO_CONFIG>")
    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    input_dir = config["path_to_intermediate_results"]
    output_dir = config["path_to_intermediate_results"]
    nerd = config["nerd"]
    benchmark = config["benchmark"]
    fs_max_evidences = config["fs_max_evidences"]
    topg = config["top-gst-number"]

    if function == "--train":
        fs = CreateDatasetTemporalFactScoring(config)
        part = function.split("-")[-1]
        input_path = os.path.join(input_dir, benchmark, nerd, f"train-ers-{fs_max_evidences}-gst-{topg}.jsonl")
        output_path = os.path.join(input_dir, benchmark, nerd, f"temporal-train.csv")
        fs.create_train_data(input_path, output_path)

    elif function == "--dev":
        fs = CreateDatasetTemporalFactScoring(config)
        input_path = os.path.join(input_dir, benchmark, nerd, f"dev-ers-{fs_max_evidences}-gst-{topg}.jsonl")
        output_path = os.path.join(input_dir, benchmark, nerd, f"temporal-dev.csv")
        fs.create_train_data(input_path, output_path)