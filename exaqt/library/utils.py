import os
import sys
import yaml
import json
import logging
import re
import hashlib
from pathlib import Path

TIMESTAMP_PATTERN_1 = re.compile('^"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z"')
TIMESTAMP_PATTERN_2 = re.compile("^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z")
TIMESTAMP_PATTERN_3 = re.compile("^[-][0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z")
def is_timestamp(string):
    """Return if the given string is a timestamp."""
    if TIMESTAMP_PATTERN_1.match(string.strip()) or TIMESTAMP_PATTERN_2.match(
            string.strip()) or TIMESTAMP_PATTERN_3.match(string.strip()):
        return True
    else:
        return False
def convert_number_to_month(number):
    """Map the given month to a number."""
    return {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }[number]

def convert_timestamp_to_date(timestamp):
    """Convert the given timestamp to the corresponding date."""
    try:
        adate = timestamp.rsplit("-", 2)
        # parse data
        year = adate[0]
        month = convert_number_to_month(adate[1])
        day = adate[2].split("T")[0]
        # remove leading zero
        if day[0] == "0":
            day = day[1]
        if day == "1" and adate[1] == "01":
            # return year for 1st jan
            return year
        date = f"{day} {month} {year}"
        return date
    except:
        # print(f"Failure with timestamp {timestamp}")
        return timestamp
def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            #word = line.strip().decode('UTF-8')
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def get_config(path):
    """Load the config dict from the given .yml file."""
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config

def get_property(path):
    """Load the property dict from the given .json file."""
    property = {}
    with open(path) as json_data:
        datalist = json.load(json_data)
    json_data.close()
    for item in datalist:
        pro = {}
        qid = item['property']['value'].replace("http://www.wikidata.org/entity/", "")
        pro["label"] = item['propertyLabel']['value']
        pro["type"] = item['propertyType']['value']
        if "propertyAltLabel" in item:
            pro["altLabel"] = item['propertyAltLabel']['value']
        else:
            pro["altLabel"] = ""
        property[qid] = pro
    return property

def get_clocq_name_type_fact_dic(wiki_ids_facts):
    PRE_PATTERN = re.compile('^P[0-9]+$')
    evidences = []
    for id_facts, retrieve_entity in wiki_ids_facts:
        if not id_facts: continue
        for fact in id_facts:
            wikidata_entities = list()
            # get md5 hash of fact
            str2hash = ''
            for it in fact:
                str2hash += it['id']
                if not PRE_PATTERN.match(it['id']):
                    if it not in wikidata_entities:
                        wikidata_entities.append(it)

            md5hash = hashlib.md5(str2hash.encode()).hexdigest()
            statementid = f"{fact[0]['id']}-{md5hash}"

            if not statementid:
                print ("error!!")
                print (fact)
            fact_spo = {'ps':[], 'pq':[]}
            fact_spo['ps'].append((fact[0], fact[1], fact[2]))
            pqn = int((len(fact) - 3) / 2)
            if pqn > 0:
                for i in range(pqn):
                    pq_fact = (fact[3 + i * 2], fact[4 + i * 2])
                    if pq_fact not in fact_spo['pq']:
                        fact_spo['pq'].append(pq_fact)

            evidence={'retrieve_for_entity':retrieve_entity, 'fact':fact, 'statement':statementid, 'fact_spo':fact_spo, 'wikidata_entities': wikidata_entities}
            evidences.append(evidence)
    return evidences


def replace_symbols(s):
    #s = s.replace('<entity>', ' ')
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace(':', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'s',' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n',' ')
    s = s.strip(',')
    s = s.strip('.')
    s = s.strip('#')
    s = s.strip('-')
    s = s.strip('\'')
    s = s.strip(';')
    s = s.strip('\"')
    s = s.strip('/')
    s = s.rstrip('?')
    s = s.rstrip('!')
    s = s.strip()
    return s


def replace_symbols_in_relation(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace(':', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('\'s', ' ')
    s = s.replace('\'', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('/', ' ')
    s = s.replace('\\', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('.', ' ')
    s = s.replace(',', ' ')
    s = s.replace('\"', ' ')
    s = s.strip()
    return s

def replace_symbols_in_entity(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.replace(':', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace("'", ' ')
    s = s.replace(';', ' ')
    s = s.replace('/', ' ')
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    s = s.replace('+', ' ')
    s = s.replace('.', ' ')
    s = s.strip('"@fr')
    s = s.strip('"@en')
    s = s.strip('"@cs')
    s = s.strip('"@de')
    s = s.strip()
    return s


def format_answers(instance):
    # TBD:  JZ need to be updated if answer format will be changed for Temp.Ans questions
    """
    Reformat answers in the TimeQuestions dataset.
    """
    _answers = list()
    for answer in instance["Answer"]:
        if answer["AnswerType"] == "Entity":
            answer = {"id": answer["WikidataQid"], "label": answer["WikidataLabel"]}
        elif answer["AnswerType"] == "Value":
            answer = {
                    "id": answer["AnswerArgument"],
                    "label": answer["AnswerArgument"]
                }
        elif answer["AnswerType"] == "Timestamp":
            answer = {
                    "id": answer["AnswerArgument"],
                    "label": answer["AnswerArgument"]
                }
        # elif answer["AnswerType"] == "Timespan":

        else:
            raise Exception
        _answers.append(answer)
    return _answers

def store_json_with_mkdir(data, output_path, indent=True):
    """Store the JSON data in the given path."""
    # create path if not exists
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        fp.write(json.dumps(data, indent=4))

def get_logger(mod_name, config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger(mod_name)
    # add handler and format
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    log_level = config["log_level"]
    logger.setLevel(getattr(logging, log_level))
    return logger


def print_dict(python_dict):
    """Print python dict as json-string."""
    json_string = json.dumps(python_dict)
    print(json_string)

def print_verbose(config, string):
    """Print the given string if verbose is set."""
    if config["verbose"]:
        print(str(string))

def get_result_logger(config):
    """Get a logger instance for the given module name."""
    # create logger
    logger = logging.getLogger("result_logger")
    # add handler and format
    method_name = config["name"]
    benchmark = config["benchmark"]
    result_file = f"_results/{benchmark}/{method_name}.res"
    result_dir = os.path.dirname(result_file)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(result_file)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # set log level
    logger.setLevel("INFO")
    return logger



