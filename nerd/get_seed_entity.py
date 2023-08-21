"""Script to get seed entities from ELQ.

Output will be a seed entity file and a wikidata qid file:
Note that the program should run under directory of BLINK-master
after building the environment of ELQ (https://github.com/facebookresearch/BLINK/tree/master/elq)
conda activate el4qa
The format of the seed entities is:
ELQ:
{'predictions': predictions, 'timing': run time of predictions}
predictions is a list of ('question id', 'question text', 'entities')
entities is a list of (qid, score, mention)
"""
import elq.main_dense as main_dense
import os
import argparse
import json
import time
import requests
import truecase
from filelock import FileLock
from pathlib import Path
import pickle
import nltk
nltk.download('punkt')
from entity_recognition_disambiguation import NERD
from library.utils import get_logger, get_result_logger
"""Script to get seed entities from WAT.
Output will be a seed entity file and a wikidata qid file:
The format of the seed entities is:
WAT:
wat_ent['spot']: list of ('spot', 'wiki_title', 'wiki_id', 'rho', 'start', 'end')
wat_ent['wikidata']: list of (wikidata_id, link[1])
"""

class TagMeAnnotation:
    # An entity annotated by WAT
    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['link_probability']
        # annotated text
        self.spot = d['spot']
        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }


class WATAnnotation:
    # An entity annotated by WAT
    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']
        # annotated text
        self.spot = d['spot']
        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }

class EntityLinkELQWATMatch(NERD):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.wat_threshold = self.config["wat_threshold"]
        self.tagme_threshold = self.config["tagme_threshold"]
        self.elq_models_path = self.config["elq_models_path"]
        self.MY_GCUBE_WAT_TOKEN = self.config["MY_GCUBE_WAT_TOKEN"]
        self.MY_GCUBE_TAGME_TOKEN = self.config["MY_GCUBE_TAGME_TOKEN"]

        self.elq_config = {
            "interactive": False,
            "biencoder_model": os.path.join(self.elq_models_path, "elq_wiki_large.bin"),
            "biencoder_config": os.path.join(self.elq_models_path, "elq_large_params.txt"),
            "cand_token_ids_path": os.path.join(self.elq_models_path, "entity_token_ids_128.t7"),
            "entity_catalogue": os.path.join(self.elq_models_path, "entity.jsonl"),
            "entity_encoding": os.path.join(self.elq_models_path, "all_entities_large.t7"),
            "output_path": "logs/",  # logging directory
            "faiss_index": "none",
            "index_path": os.path.join(self.elq_models_path, "faiss_hnsw_index.pkl"),
            "num_cand_mentions": 10,
            "num_cand_entities": 10,
            "threshold_type": "joint",
            "threshold": -4.5,
        }
        self.args = argparse.Namespace(**self.elq_config)
        self.logger.info(f"Prepare data and start...")
        self.models = main_dense.load_models(self.args, logger=None)
        self.id2wikidata = json.load(open(os.path.join(self.elq_models_path, "id2wikidata.json")))

        self.use_cache = self.config["nerd_use_cache"]
        self.cache_path = self.config["nerd_cache_path"]

        # initialize dump
        if self.use_cache:
            self._init_cache()
            self.cache_changed = False

    def get_entity_prediction(self, question):
        id = 1
        data_to_link = [{'id': id, 'text': question.strip() + '?'}]

        start = time.time()
        predictions = main_dense.run(self.args, None, *self.models, test_data=data_to_link)
        end = time.time() - start

        predictions = [{
            'id': prediction['id'],
            'text': prediction['text'],
            'entities': [(self.id2wikidata.get(prediction['pred_triples'][idx][0]), prediction['scores'][idx],
                          prediction['pred_tuples_string'][idx][1]) for idx in range(len(prediction['pred_triples']))],
        } for prediction in predictions]

        result = {'predictions': predictions, 'timing': end}
        print (f"ELQ result{result}")
        return result

    def get_seed_entities_wat(self, question):
        tagme_ent = self.get_response_wat(question)
        tagme_ent['wikidata'] = []
        for link in tagme_ent['spot']:
            pageid = link[2]
            wikipedia_link = self.get_wikipedialink(pageid)
            if wikipedia_link:
                wikidata_id = self.get_qid(wikipedia_link)
                if wikidata_id:
                    tagme_ent['wikidata'].append((wikidata_id, link[1]))
        return tagme_ent

    def get_response_wat(self, question):
        wat_ent = {}
        wat_ent['spot'] = []
        try:
            annotations = self.wat_entity_linking(question)
            # print (annotations)
            if annotations:
                for doc in annotations:
                    if doc['rho'] >= self.wat_threshold:
                        doc['spot'] = question[doc["start"]:doc["end"]]
                        wat_ent['spot'].append(
                            (doc['spot'], doc['wiki_title'], str(doc['wiki_id']), doc['rho'], doc['start'], doc['end']))
        except:
            self.logger.info(f"WAT Problem {question}.")
        return wat_ent

    def get_wikipedialink(self, pageid):
        info_url = "https://en.wikipedia.org/w/api.php?action=query&prop=info&pageids=" + pageid + "&inprop=url&format=json"
        try:
            response = requests.get(info_url)
            result = response.json()["query"]["pages"]
            # print(result)
            if result:
                link = result[pageid]['fullurl']
                return link
        except:
            self.logger.info(f"get_wikipedialink problem {pageid}.")

    def get_qid(self, wikipedia_link):
        url = f"https://openrefine-wikidata.toolforge.org/en/api?query={wikipedia_link}"
        try:
            response = requests.get(url)
            results = response.json()["result"]
            if results:
                qid = results[0]['id']
                wikidata_label = results[0]['name']
                return qid, wikidata_label
        except:
            self.logger.info(f"get_qid problem {wikipedia_link}.")

    def tagme_entity_linking(self, text):
        # Main method, text annotation with WAT entity linking system
        tagme_url = f'https://tagme.d4science.org/tagme/tag?lang=en&gcube-token={self.MY_GCUBE_TAGME_TOKEN}&text={text}'
        try:
            response = requests.get(tagme_url)
            tagme_annotations = [TagMeAnnotation(a) for a in response.json()['annotations']]
            return [w.json_dict() for w in tagme_annotations]
        except:
            self.logger.info(f"here is a timeout error!")
            return None

    def get_seed_entities_tagme(self, question):
        tagme_ent = self.get_response_tagme(question)
        tagme_ent['wikidata'] = []
        for link in tagme_ent['spot']:
            pageid = link[2]
            wikipedia_link = self.get_wikipedialink(pageid)
            if wikipedia_link:
                try:
                    wikidata_id, wikidata_label = self.get_qid(wikipedia_link)
                    if wikidata_id:
                        tagme_ent['wikidata'].append((wikidata_id, wikidata_label, link[1]))
                except:
                    self.logger.info(f"get_qid problem {wikipedia_link}.")
                    continue
        return tagme_ent

    def get_response_tagme(self, question):
        tagme_ent = {}
        tagme_ent['spot'] = []
        try:
            annotations = self.tagme_entity_linking(question)
            # print (annotations)
            if annotations:
                for doc in annotations:
                    if doc['rho'] >= self.tagme_threshold:
                        doc['spot'] = question[doc["start"]:doc["end"]]
                        tagme_ent['spot'].append(
                            (doc['spot'], doc['wiki_title'], str(doc['wiki_id']), doc['rho'], doc['start'], doc['end']))
        except:
            self.logger.info(f"TAGME Problem {question}.")
        return tagme_ent

    def wat_entity_linking(self, text):
        # Main method, text annotation with WAT entity linking system
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", self.MY_GCUBE_WAT_TOKEN),
                   ("text", text),
                   ("lang", 'en'),
                   ("tokenizer", "nlp4j"),
                   ('debug', 9),
                   ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]
        try:
            response = requests.get(wat_url, params=payload)
            wat_annotations = [WATAnnotation(a) for a in response.json()['annotations']]
            return [w.json_dict() for w in wat_annotations]
        except:
            self.logger.info(f"timeout error!")
            return None

    def inference_on_instance(self, instance):
        question = instance["Question"]
        if self.use_cache and question in self.cache:
            result_dic = self.cache[question]
            instance.update(result_dic)
        else:
            question_truecase = truecase.get_true_case(question)
            elq_result = self.get_entity_prediction(question_truecase)
            wat_ent = self.get_seed_entities_wat(question_truecase)
            tagme_ent = self.get_seed_entities_tagme(question_truecase)
            wiki_ids_elq = list()
            wiki_ids_wat = list()
            wiki_ids_tagme = list()
            elq_dic = {}
            for result in elq_result["predictions"]:
                elq_dic[str(result['id'])] = result["entities"]

            for (qid, score, text) in elq_dic[str(result['id'])]:
                if qid is not None:
                    wiki_ids_elq.append([qid, score, text])

            if 'wikidata' in wat_ent and 'spot' in wat_ent:
                for id1 in wat_ent['wikidata']:
                    index = wat_ent['wikidata'].index(id1)
                    text = wat_ent['spot'][index][0].lower()
                    score = float(wat_ent['spot'][index][3])
                    wiki_ids_wat.append([id1[0], score, text])

            if 'wikidata' in tagme_ent and 'spot' in tagme_ent:
                for id1 in tagme_ent['wikidata']:
                    index = tagme_ent['wikidata'].index(id1)
                    text = tagme_ent['spot'][index][0].lower()
                    score = float(tagme_ent['spot'][index][3])
                    wiki_ids_tagme.append([id1[0], score, text, id1[1]])

            result_dic = {"tagme": wiki_ids_tagme, "elq":wiki_ids_elq, "wat": wiki_ids_wat}
            instance.update(result_dic)

            if self.use_cache:
                self.cache_changed = True
                self.cache[question] = result_dic

    def store_cache(self):
        """Store the cache to disk."""
        if not self.use_cache:  # store only if cache in use
            return
        if not self.cache_changed:  # store only if cache changed
            return
        # check if the cache was updated by other processes
        if self._read_cache_version() == self.cache_version:
            # no updates: store and update version
            self.logger.info(f"Writing NERD cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self._write_cache(self.cache)
                self._write_cache_version()
        else:
            # update! read updated version and merge the caches
            self.logger.info(f"Merging NERD cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                # read updated version
                updated_cache = self._read_cache()
                # overwrite with changes in current process (most recent)
                updated_cache.update(self.cache)
                # store
                self._write_cache(updated_cache)
                self._write_cache_version()

    def _init_cache(self):
        """Initialize the cache."""
        if os.path.isfile(self.cache_path):
            # remember version read initially
            self.logger.info(f"Loading NERD cache from path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self.cache_version = self._read_cache_version()
                self.logger.debug(self.cache_version)
                self.cache = self._read_cache()
            self.logger.info(f"NERD cache successfully loaded.")
        else:
            self.logger.info(f"Could not find an existing NERD cache at path {self.cache_path}.")
            self.logger.info("Populating NERD cache from scratch!")
            self.cache = {}
            self._write_cache(self.cache)
            self._write_cache_version()

    def _read_cache(self):
        """
        Read the current version of the cache.
        This can be different from the version used in this file,
        given that multiple processes may access it simultaneously.
        """
        # read file content from cache shared across QU methods
        with open(self.cache_path, "rb") as fp:
            cache = pickle.load(fp)
        return cache

    def _write_cache(self, cache):
        """Write to the cache."""
        cache_dir = os.path.dirname(self.cache_path)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fp:
            pickle.dump(cache, fp)
        return cache

    def _read_cache_version(self):
        """Read the cache version (hashed timestamp of last update) from a dedicated file."""
        if not os.path.isfile(f"{self.cache_path}.version"):
            self._write_cache_version()
        with open(f"{self.cache_path}.version", "r") as fp:
            cache_version = fp.readline().strip()
        return cache_version

    def _write_cache_version(self):
        """Write the current cache version (hashed timestamp of current update)."""
        with open(f"{self.cache_path}.version", "w") as fp:
            version = str(time.time())
            fp.write(version)
        self.cache_version = version
