"""Script to read seed entities of each question and extract triples for the seed entities.

In the code of EXAQT at CIKM2021, the output is a txt SPO file for each question with the following format:
statement_id-ps:||subject_qid||label||ps_qid||ps_qid||object_qid||label
statement_id-pq:||subject_qid||label||ps_qid||pq_qid||object_qid||label

For decreasing the number of files, we create one json file for each dataset right now.
In the json file, for each fact (called evidence) of each question, you can find it's SPO format content in the key
"statement_spo_list", which is the same as the format of the previous SPO file.
"""
import re

from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
from exaqt.library.utils import get_logger, get_clocq_name_type_fact_dic

class FactRetriever:
    def __init__(self, config, property):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.property = property
        # initialize clocq for KB-facts
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

    def two_hop_temporal_fact_retriever(self, ids):
        wiki_ids_facts = []
        for id in ids:
            facts = self.clocq.get_neighborhood_two_hop(id, include_labels=True, p=10000)
            wiki_ids_facts.append((facts, id))
        two_hop_evidences = get_clocq_name_type_fact_dic(wiki_ids_facts)
        tempfact_evidences = self._filter_non_temporal(two_hop_evidences)
        self._convert_tempfact_to_spo(tempfact_evidences, ids)
        return tempfact_evidences

    def _filter_non_temporal(self, evidences):
        temporal_evidences = []
        for evidence in evidences:
            fact_spo = evidence["fact_spo"]
            istemp = 0
            for item in fact_spo['ps']:
                if item[1]['id'] in self.property:
                    if "http://wikiba.se/ontology#Time" in self.property[item[1]['id']]["type"]:
                        istemp = 1
                        break
            for item in fact_spo['pq']:
                if item[0]['id'] in self.property:
                    if "http://wikiba.se/ontology#Time" in self.property[item[0]['id']]["type"]:
                        istemp = 1
                        break
            if istemp == 1:
                temporal_evidences.append(evidence)
        return temporal_evidences

    def _convert_tempfact_to_spo(self, temporal_evidences, wiki_ids):
        #evidence={'fact':fact, 'statement':statementid, 'fact_spo':fact_spo, 'wikidata_entities': wikidata_entities}
        # for entities
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        for evidence in temporal_evidences:
            statement_spo_list = list()
            statement = evidence["statement"]
            fact_dic = evidence["fact_spo"]

            t = fact_dic['ps'][0]
            sub = t[0]['id'].strip('"')
            pre = t[1]['id'].strip('"')
            obj = t[2]['id'].strip('"')
            subname = sub
            objname = obj

            if isinstance(t[0]['label'], list):
                for item in t[0]['label']:
                    if ENT_PATTERN.match(item) == None:
                        subname = item
                        break
            else:
                subname = t[0]['label'].strip('"')
            if isinstance(t[2]['label'], list):
                for item in t[2]['label']:
                    if ENT_PATTERN.match(item) == None:
                        objname = item
                        break
            else:
                objname = t[2]['label'].strip('"')
            for id in wiki_ids:
                if sub == id:
                    sub = "corner#" + sub
                if obj == id:
                    obj = "corner#" + obj

            p = sub + "||" + subname + "||" + pre
            statement_spo_list.append(f"{statement}-ps:||{p}||{pre}||{obj}||{str(objname)}")

            for pqt in fact_dic['pq']:
                pre = pqt[0]['id'].strip('"')
                obj = pqt[1]['id'].strip('"')
                objname = obj
                if isinstance(pqt[1]['label'], list):
                    for item in pqt[1]['label']:
                        if ENT_PATTERN.match(item) == None:
                            objname = item
                            break
                else:
                    objname = pqt[1]['label'].strip('"')
                for id in wiki_ids:
                    if obj == id:
                        obj = "corner#" + obj
                pq_line = f"{statement}-pq:||{p}||{pre}||{obj}||{objname}"
                if pq_line not in statement_spo_list:
                    statement_spo_list.append(pq_line)

            evidence["statement_spo_list"] = statement_spo_list
            if evidence.get("fact_spo"):
                del evidence["fact_spo"]
        return temporal_evidences

    def temporal_fact_retriever(self, wiki_ids):
        wiki_ids_facts = []
        for id in wiki_ids:
            facts = self.clocq.get_neighborhood(id, include_labels=True, p=10000)
            wiki_ids_facts.append((facts, id))
        evidences = get_clocq_name_type_fact_dic(wiki_ids_facts)
        tempfact_evidences = self._filter_non_temporal(evidences)
        self._convert_tempfact_to_spo(tempfact_evidences, wiki_ids)
        return tempfact_evidences

    def get_wikidata_tuplesfromclocq(self, wiki_ids):
        done = set()
        wiki_ids_facts = []
        for ids in wiki_ids:
            id1 = ids[0]
            if id1 in done:
                continue
            # score = ids[1]
            # text = ids[2]
            id1 = id1.strip()
            id2 = id1.lstrip('http://www.wikidata.org/entity/')
            # print("Expand Id")
            # print(id1, id2, score, text)
            facts = self.clocq.get_neighborhood(id2, include_labels=True, p=10000)
            wiki_ids_facts.append((facts, id2))
            done.add(id1)
        return get_clocq_name_type_fact_dic(wiki_ids_facts)

    def fact_retriever(self, wiki_ids):
        evidences = self.get_wikidata_tuplesfromclocq(wiki_ids)
        # for entities
        ENT_PATTERN = re.compile('^Q[0-9]+$')

        for evidence in evidences:
            statement_spo_list = list()
            statement = evidence["statement"]
            fact_dic  = evidence["fact_spo"]

            t = fact_dic['ps'][0]
            sub = t[0]['id'].strip('"')
            pre = t[1]['id'].strip('"')
            obj = t[2]['id'].strip('"')
            subname = sub
            objname = obj

            if isinstance(t[0]['label'], list):
                for item in t[0]['label']:
                    if ENT_PATTERN.match(item) == None:
                        subname = item
                        break
            else:
                subname = t[0]['label'].strip('"')
            if isinstance(t[2]['label'], list):
                for item in t[2]['label']:
                    if ENT_PATTERN.match(item) == None:
                        objname = item
                        break
            else:
                objname = t[2]['label'].strip('"')
            for ids in wiki_ids:
                id1 = ids[0]
                score = ids[1]
                text = ids[2]
                id2 = id1.lstrip('http://www.wikidata.org/entity/')
                if sub == id2:
                    sub = "corner#" + sub + "#" + str(score) + "#" + text
                if obj == id2:
                    obj = "corner#" + obj + "#" + str(score) + "#" + text

            p = sub + "||" + subname + "||" + pre
            statement_spo_list.append(f"{statement}-ps:||{p}||{pre}||{obj}||{str(objname)}")

            for pqt in fact_dic['pq']:
                pre = pqt[0]['id'].strip('"')
                obj = pqt[1]['id'].strip('"')
                objname = obj
                if isinstance(pqt[1]['label'], list):
                    for item in pqt[1]['label']:
                        if ENT_PATTERN.match(item) == None:
                            objname = item
                            break
                else:
                    objname = pqt[1]['label'].strip('"')
                for ids in wiki_ids:
                    id1 = ids[0]
                    score = ids[1]
                    text = ids[2]
                    id2 = id1.lstrip('http://www.wikidata.org/entity/')
                    if obj == id2:
                        obj = "corner#" + obj + "#" + str(score) + "#" + text
                statement_spo_list.append(f"{statement}-pq:||{p}||{pre}||{obj}||{objname}")

            evidence["statement_spo_list"] = statement_spo_list
            if evidence.get("fact_spo"):
                del evidence["fact_spo"]

        return evidences

    def twohop_fact_retriever(self, wiki_ids):
        #graftnet
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        nerd_entities = [ids[0] for ids in wiki_ids]
        twohop_facts = {nerd_entity: [] for nerd_entity in nerd_entities}
        twohop_entities = {nerd_entity: [] for nerd_entity in nerd_entities}
        self.logger.info(f"Retrieve for nerd entities: {len(nerd_entities)}.")
        search = {}
        for wiki_id in wiki_ids:
            evidences = self.fact_retriever([wiki_ids])
            #self.logger.info(f"Done with one hop evidence retriever: {len(evidences)}.")
            neighbor_entities = []
            for evidence in evidences:
                twohop_facts[wiki_id] = evidence["statement_spo_list"]
                if len(evidence["fact"]) == 3:
                    onehop_neighbor_entities = [[item["id"], '0.0', 'null'] for item in evidence["fact"] if
                    ENT_PATTERN.match(item["id"]) and item["id"] not in nerd_entities]
                    twohop_entities[wiki_id] = [item[0] for item in onehop_neighbor_entities]
                    neighbor_entities += onehop_neighbor_entities
                    for item in twohop_entities[wiki_id]:
                        if item not in search:
                            search[item] = []
                        search[item].append(wiki_id)

            self.logger.info(f"Retrieve for neighbors of nerd entities: {len(neighbor_entities)}.")
            twohop_evidences = self.fact_retriever(neighbor_entities)

            #twohop_evidences = self.fact_retriever(neighbor_entities)
            for evidence in twohop_evidences:
                if len(evidence["fact"]) == 3:
                    for item in search[evidence["retrieve_for_entity"]]:
                        twohop_facts[item] += evidence["statement_spo_list"]

        self.logger.info(f"Done with two hop evidence retriever: {len(twohop_facts)}.")
        return twohop_facts

    def twohop_fact_retriever_previous_version(self, wiki_ids):
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        nerd_entities = [ids[0] for ids in wiki_ids]
        self.logger.info(f"Retrieve for nerd entities: {len(nerd_entities)}.")
        evidences = self.fact_retriever(wiki_ids)
        self.logger.info(f"Done with one hop evidence retriever: {len(evidences)}.")
        neighbor_entities = []
        for evidence in evidences:
            if len(evidence["fact"]) == 3:
                neighbor_entities += [[item["id"], '0.0', 'null'] for item in evidence["fact"] if
                                      ENT_PATTERN.match(item["id"]) and item["id"] not in nerd_entities]

        self.logger.info(f"Retrieve for neighbors of nerd entities: {len(neighbor_entities)}.")
        twohop_evidences = self.fact_retriever(neighbor_entities)
        for evidence in twohop_evidences:
            if len(evidence["fact"]) == 3:
                evidences.append(evidence)

        for evidence in evidences:
            del evidence['fact']
            del evidence['retrieve_for_entity']
            del evidence['statement']

        self.logger.info(f"Done with two hop evidence retriever: {len(evidences)}.")
        return evidences
