###Construct dictionaries including words, dates, entities, relations, categories, signals
# from questions and subgraphs of train, dev and test dataset

import json

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

def is_digit(z):
    try:
        z=int(z)
        return isinstance(z,int)
    except ValueError:
        return False

def get_dictionary(in_files):
    entities = set()
    #clique_entities = set()
    tempentities = list()
    relations = list()
    trelations = list()
    datas = list()
    categories = list()
    signals = list()
    entityname = dict()
    words = list()

    #words.append('__unk__')
    for file in in_files:
        with open(file, "r") as f:
            for line in f:
                try:
                    datas.append(json.loads(line.strip()))
                except:
                    print ("\nThere is an error!")
                    print (line.strip())
                    continue

        print ("Number of questions:", str(len(datas)))
    for data in datas:
        words += [replace_symbols(item) for item in data['question'].strip().split()]
        categories += data["type"]
        signals += data["signal"]
        tempentities += data["tempentities"]
        trelations += data["temprelations"]
        entityname.update(data["entityname"])
        for entity in data["subgraph"]["entities"]:
            entities.add(entity["kb_id"])
        for entity in data["entities"]:
            entities.add(entity["kb_id"])

        for tem in data["tempentities"]:
            entities.add(tem)

        for tuple in data["subgraph"]["tuples"]:
            # rel = list()
            for item in tuple:
                if "rel_id" in item:
                    rel_name = item["rel_id"]
                    relations.append(rel_name)
                if "kb_id" in item:
                    entity = item["kb_id"]
                    entities.add(entity)

    return words, list(entities), list(set(relations)), list(set(trelations)), list(set(tempentities)), categories, signals, entityname
