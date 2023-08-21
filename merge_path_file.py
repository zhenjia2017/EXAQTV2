import os
import json
from tqdm import tqdm

directory = r"C:\Users\zhenj\PycharmProjects\EXAQT\_intermediate_representations\timplicitquestions\path-part"

file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
output = os.path.join(directory, "train-ers-25-path.jsonl")
results = []
for file_name in file_list:
    print(file_name)

    with open(os.path.join(directory, file_name), "r") as fp:
        for line in tqdm(fp):
            instance = json.loads(line)
            if instance not in results:
                results.append(instance)

sorted_list = sorted(results, key=lambda x: x['Id'])

with open(output, "w") as fout:
    for instance in sorted_list:
        fout.write(json.dumps(instance))
        fout.write("\n")


