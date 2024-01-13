import time
import re
from exaqt.library.utils import get_logger
from exaqt.answer_graph.answer_graph_construction import AnswerGraph
from exaqt.answer_graph.fact_retriever.fact_er import FactRetriever
from exaqt.answer_graph.fact_scoring.fact_scoring_module import FactScoringModule
from exaqt.answer_graph.fact_scoring.temporalfact_scoring_module import TempFactScoringModule
from exaqt.answer_graph.connectivity.seed_path_extractor import SeedPathExtractor
from exaqt.answer_graph.gst_construction.compact_subgraph_constructor import QuestionUnionGST

class FTRS(AnswerGraph):
    """
    Extract fact, score, GST construction, temporal enhance and temporal facts scores.
    """
    def __init__(self, config, property):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.ftr = FactRetriever(config, property)
        self.ftc = SeedPathExtractor(config)
        self.fts = FactScoringModule(config, property)
        self.temfs = TempFactScoringModule(config, property)
        self.gst = QuestionUnionGST(config, property)
        self.nerd = self.config["nerd"]

    def inference_on_instance(self, instance):
        """Retrieve fact."""
        start = time.time()
        self.er_inference_on_instance(instance)
        self.logger.debug(f"Time taken (Fact Retrieval): {time.time() - start} seconds")
        self.logger.debug(
            f"Time taken (Fact Retrieval, Seed Connectivity): {time.time() - start} seconds")
        self.ers_inference_on_instance(instance)
        self.path_inference_on_instance(instance)
        self.gst_inference_on_instance(instance)
        self.tempers_inference_on_instance(instance)
        self.logger.debug(f"Time taken (Fact Retrieval and Fact Scoring): {time.time() - start} seconds")

    def path_inference_on_instance(self, instance):
        connectivity_result = self.ftc.seed_pairs_best_path(instance)
        instance["connectivity"] = connectivity_result

    def connectivity_check_inference_on_instance(self, instance):
        connectivity_result = self.ftc.seed_connectivity_path(instance)
        return connectivity_result

    def er_inference_on_instance(self, instance):
        wiki_ids = list()
        wiki_ids = instance["elq"]
        if self.nerd == "elq-wat":
            wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
        elif self.nerd == "elq-tagme":
            wiki_ids += instance["tagme"]
        elif self.nerd == "elq-tagme-wat":
            wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
            wiki_ids += instance["tagme"]

        evidences = self.ftr.fact_retriever(wiki_ids)
        instance["candidate_evidences"] = evidences
        return evidences

    def ers_inference_on_instance(self, instance):
        top_evidences = self.fts.get_top_evidences(instance)
        instance["candidate_evidences"] = top_evidences
        return top_evidences

    def tempers_inference_on_instance(self, instance):
        top_evidences = self.temfs.get_top_evidences(instance)
        instance["temporal_evidences"] = top_evidences
        return top_evidences

    def gst_inference_on_instance(self, instance):
        entity_weight, gst_spo, cornerstone, gst_entities = self.gst.get_gst_for_instance(instance)
        instance["entity_weight"] = entity_weight
        instance["complete_gst_spo_list"] = gst_spo
        instance["cornerstone"] = cornerstone
        instance["complete_gst_entity"] = gst_entities
        ENT_PATTERN = re.compile('^Q[0-9]+$')
        if instance["complete_gst_entity"]:
            completedgst_can_entities_qids = [item["id"] for item in instance["complete_gst_entity"] if
                                          ENT_PATTERN.match(item["id"]) != None]
            instance["temporal_evidences"] = self.ftr.temporal_fact_retriever(completedgst_can_entities_qids)
        else:
            instance["temporal_evidences"] = []

    def train(self):
        self.fts.train()

    def temporal_train(self):
        self.temfs.train()
