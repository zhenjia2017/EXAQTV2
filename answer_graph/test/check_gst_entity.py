import os
import sys
from tqdm import tqdm
import json
from exaqt.library.utils import get_config, get_logger, get_result_logger
from exaqt.evaluation import answer_presence, answer_presence_gst

def get_extractentities_spo(spos):
    spo_entity = list()
    # if os.path.exists(spo_file):
    #     print(spo_file + ' exists.')
    # else:
    #     print(spo_file + ' not found!')
    #     return spo_entity
    # f22 = open(spo_file, 'r')
    # spos = f22.readlines()
    # f22.close()
    for line in spos:
        triple = line.strip().split('||')
        if len(triple) < 7 or len(triple) > 7: continue
        sub = triple[1]
        obj = triple[5]
        sub_name = triple[2]
        obj_name = triple[6]
        if 'corner#' in sub:
            sub = sub.replace('corner#', '').split('#')[0]
        if 'corner#' in obj:
            obj = obj.replace('corner#', '').split('#')[0]
        if {"id": sub, "label":sub_name} not in spo_entity:
            spo_entity.append({"id": sub, "label":sub_name})
        if {"id": obj, "label":obj_name} not in spo_entity:
            spo_entity.append({"id": obj, "label":obj_name})

    return spo_entity

class CheckGSTAnswerRecall:
    def __init__(self, config):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.name = self.config["name"]
        self.nerd = self.config["nerd"]
        self.fs_max_evidences = self.config["fs_max_evidences"]
        self.topg = self.config["top-gst-number"]

    def check(self, dataset):
        input_dir = self.config["path_to_intermediate_results"]
        output_dir = self.config["path_to_intermediate_results"]

        # process data
        input_path = os.path.join(input_dir, self.nerd, f"{dataset}-ers-{self.fs_max_evidences}-gst-{self.topg}.jsonl")

        count1 = 0
        count2 = 0
        answer_presence_nodes = []
        answer_presence_spos = []
        with open(input_path, 'r') as fi:
            for line in tqdm(fi):
                # line in fi:
                count2 += 1
                instance = json.loads(line)
                gst_spo = instance["complete_gst_spo_list"]
                #completedgst_can_entities = [{"id": node.split("::")[2], "label":node.split("::")[0]} for node in completedGST.nodes() if node.split("::")[1] == "Entity"]
                gst_entities = instance["complete_gst_entity"]
                spo_entity = get_extractentities_spo(gst_spo)
                hit_nodes = answer_presence_gst(gst_entities, instance["answers"])
                hit_spos = answer_presence_gst(spo_entity, instance["answers"])
                answer_presence_nodes += [hit_nodes]
                answer_presence_spos += [hit_spos]
                if answer_presence_nodes != answer_presence_spos:
                    print (instance["Id"])
                    count1 += 1

            # print results
            print (f"evaluation answer_presence_nodes result:\n")
            print (sum(answer_presence_nodes) / len(answer_presence_nodes))
            print (f"\nevaluation answer_presence_spos result:\n")
            print (sum(answer_presence_spos) / len(answer_presence_spos))


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Usage: python exaqt/answer_graph/test/check_gst_entity.py <DATASET> <PATH_TO_CONFIG>")

    # load config
    dataset = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    check = CheckGSTAnswerRecall(config)
    check.check(dataset)





