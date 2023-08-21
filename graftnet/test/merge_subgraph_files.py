import os
import sys
from tqdm import tqdm
import json
from exaqt.library.utils import get_config, get_logger, get_result_logger

class Merge:
    def __init__(self, config):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.name = self.config["name"]
        self.nerd = self.config["nerd"]

    def merge(self, dataset):
        input_dir = self.config["path_to_intermediate_results"]
        input_subgraph_path_1 = os.path.join(input_dir, self.name, self.nerd, f"{dataset}-subgraph.json")
        input_subgraph_path_2 = os.path.join(input_dir, self.name, self.nerd, f"{dataset}-subgraph-append.json")
        output_subgraph_1 = os.path.join(input_dir, self.name, self.nerd, f"{dataset}-subgraph-remove-ordinal.json")
        output_subgraph_2 = os.path.join(input_dir, self.name, self.nerd, f"{dataset}-subgraph-all-type.json")

        count1 = 0
        count2 = 0
        with open(output_subgraph_1, "wb") as fo1, open(output_subgraph_2, "wb") as fo2:
            with open(input_subgraph_path_1, 'r') as fi:
                for line in tqdm(fi):
                    # line in fi:
                    instance = json.loads(line)
                    categories = instance["type"]
                    fo2.write(json.dumps(instance).encode("utf-8"))
                    fo2.write("\n".encode("utf-8"))
                    count2 += 1
                    # drop ordinal
                    if "Ordinal" not in categories:
                        count1 += 1
                        fo1.write(json.dumps(instance).encode("utf-8"))
                        fo1.write("\n".encode("utf-8"))
            with open(input_subgraph_path_2, 'r') as fi:
                for line in tqdm(fi):
                    # line in fi:
                    instance = json.loads(line)
                    categories = instance["type"]
                    fo2.write(json.dumps(instance).encode("utf-8"))
                    fo2.write("\n".encode("utf-8"))
                    count2 += 1
                    # drop ordinal
                    if "Ordinal" not in categories:
                        count1 += 1
                        fo1.write(json.dumps(instance).encode("utf-8"))
                        fo1.write("\n".encode("utf-8"))

        print (count1)
        print (count2)
        category_to_subgraph_ans_pres = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}
        category_to_twohopfact_ans_pres = {"explicit": [], "implicit": [], "temp.ans": [], "all": []}
        with open(output_subgraph_1, 'r') as fi:
            for line in tqdm(fi):
                # line in fi:
                instance = json.loads(line)
                category_slot = [cat.lower() for cat in instance["type"]]
                if "ordinal" in category_slot: continue
                answer_presence = instance["answer_presence"]
                twohopfact_answer_presence = instance["twohopfact_answer_presence"]
                if answer_presence > 0:
                    category_to_subgraph_ans_pres["all"] += [True]
                else:
                    category_to_subgraph_ans_pres["all"] += [False]
                category_to_twohopfact_ans_pres["all"] += [twohopfact_answer_presence]

                for category in category_to_twohopfact_ans_pres.keys():
                    if category in category_slot:
                        category_to_twohopfact_ans_pres[category] += [twohopfact_answer_presence]
                        if answer_presence > 0:
                            category_to_subgraph_ans_pres[category] += [True]
                        else:
                            category_to_subgraph_ans_pres[category] += [False]


            # print results
            res_path = output_subgraph_1.replace(".json", ".res")
            with open(res_path, "w") as fp:
                fp.write(f"evaluation two-hop fact result:\n")
                category_answer_presence_per_src = {
                category: (sum(num) / len(num)) for category, num in category_to_twohopfact_ans_pres.items() if len(num) != 0
            }
                fp.write(f"\nCategory two-hop fact Answer presence per source: {category_answer_presence_per_src}")

                fp.write(f"\nevaluation subgraph result:\n")
                category_answer_presence_per_src = {
                    category: (sum(num) / len(num)) for category, num in category_to_subgraph_ans_pres.items() if
                    len(num) != 0
                }
                fp.write(f"\nCategory Subgraph Answer presence per source: {category_answer_presence_per_src}")

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("Usage: python exaqt/graft/test/merge_subgraph_files.py <DATASET> <PATH_TO_CONFIG>")

    # load config
    dataset = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    merge = Merge(config)
    merge.merge(dataset)