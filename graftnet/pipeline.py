import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import json
import time
import pickle
import numpy as np
from exaqt.evaluation import answer_presence
from exaqt.library.utils import get_config, get_logger, get_result_logger, replace_symbols, get_property, format_answers
from exaqt.answer_graph.fact_retriever.fact_er import FactRetriever
from exaqt.graftnet.relation_question_embeddings import RelationQuestionEmbedding
from exaqt.graftnet.pretrained_embedding_wiki2vec import relation_emb, word_emb, entity_emb
from exaqt.graftnet.get_dictionaries import get_dictionary
from exaqt.graftnet.subgraph import SubGraphGenerator
from exaqt.graftnet.main_graftnet_baseline_500_wiki2vec import train, test, data_load
from exaqt.graftnet.util import load_dict
from exaqt.answer_predict.script_listscore import compare_pr, evaluate_result_for_category

class Pipeline:
	def __init__(self, config):
		"""Create the pipeline based on the config."""
		# load config
		self.config = config
		self.logger = get_logger(__name__, config)
		self.result_logger = get_result_logger(config)
		self.property_path = self.config["pro-info-path"]
		self.property = self._load_property()
		self.ftr = FactRetriever(self.config, self.property)
		self.embeddings = RelationQuestionEmbedding(config, self.property)
		self.question_emb_pkl_file = self.config["question_wikiemb_path"]
		self.relation_emb_pkl_file = self.config["relation_wikiemb_path"]
		self.subgraph = SubGraphGenerator(config, self.property)
		self.wiki2vec = self.embeddings.wiki2vec
		# load individual modules
		self.name = self.config["name"]
		self.nerd = self.config["nerd"]
		loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
		print("Loggers", loggers)

	def _relation_question_embeddings(self):
		self.logger.info("Generate question and relation embeddings file")
		self.embeddings.generate_embeddings()

	def _generate_subgraph(self, dataset):
		input_dir = self.config["path_to_intermediate_results"]
		input_path = os.path.join(input_dir, f"{dataset}-nerd.json")
		output_path = os.path.join(input_dir, self.name, self.nerd, f"{dataset}-subgraph.json")
		# open data
		with open(input_path, "r") as fp:
			data = json.load(fp)
		self.logger.info(f"Input data loaded from: {input_path}.")

		seed_map = {}
		answer_recall = []
		twohop_answer_recall = []
		bad_questions = []
		ok_questions = []
		num_empty_tuples = 0

		with open(output_path, "wb") as fo:
			for instance in tqdm(data):
				categories = instance["Temporal question type"]
				if "Ordinal" in categories: continue
				evidences = self.er_inference_on_instance(instance)
				instance["candidate_evidences"] = evidences
				instance["answers"] = format_answers(instance)
				hit, answering_evidences = answer_presence(evidences, instance["answers"])
				instance["answer_presence"] = hit
				twohop_answer_recall += [hit]
				subgraph, num_empty_tuple = self.subgraph.generate_subgraph_instance(instance, seed_map,
																					 bad_questions, ok_questions,
																					 answer_recall)
				num_empty_tuples += num_empty_tuple

				if dataset == "train":
					if subgraph["id"] in ok_questions:
						fo.write(json.dumps(subgraph).encode("utf-8"))
						fo.write("\n".encode("utf-8"))
				else:
					fo.write(json.dumps(subgraph).encode("utf-8"))
					fo.write("\n".encode("utf-8"))

			self.logger.info("\n%d questions with empty subgraphs." % num_empty_tuples)
			self.logger.info("\nOK questions = %.3f" % (len(ok_questions) * 1.0 / len(data)))
			self.logger.info("\nNumber of OK questions = %s" % str(len(ok_questions)))
			self.logger.info("\nAnswer recall = %.3f" % (sum(answer_recall) / len(answer_recall)))
			self.logger.info("\nTwohop facts answer recall = %.3f" % (sum(twohop_answer_recall) / len(twohop_answer_recall)))

	def _generate_subgraph_train(self):
		self._generate_subgraph("train")

	def _generate_subgraph_dev(self):
		self._generate_subgraph("dev")

	def _generate_subgraph_test(self):
		self._generate_subgraph("test")

	def gcn_model_train(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = os.path.join(input_dir, self.name, self.nerd)

		result_path = os.path.join(output_dir, 'result')
		model_path = os.path.join(output_dir, 'model')
		os.makedirs(result_path, exist_ok=True)


		train_subg = os.path.join(output_dir, "train-subgraph.json")
		dev_subg = os.path.join(output_dir, "dev-subgraph.json")
		test_subg = os.path.join(output_dir, "test-subgraph.json")

		entity_dic = os.path.join(output_dir, "entities.txt")
		relation_dic = os.path.join(output_dir, "relations.txt")
		word_dic = os.path.join(output_dir, "vocab.txt")
		entity2id = load_dict(entity_dic)
		word2id = load_dict(word_dic)
		relation2id = load_dict(relation_dic)

		relation_emb_file = os.path.join(output_dir, "relations.npy")
		vocab_emb_file = os.path.join(output_dir, "vocab.npy")
		entity_emb_file = os.path.join(output_dir, "entities.npy")

		train_data, valid_data, test_data = data_load(self.config, train_subg, dev_subg, test_subg, entity2id, word2id, relation2id)
		dev_acc = train(self.config, train_data, valid_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file)

	def gcn_model_inference(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = os.path.join(input_dir, self.name, self.nerd)

		result_path = os.path.join(output_dir, 'result')

		os.makedirs(result_path, exist_ok=True)


		train_subg = os.path.join(output_dir, "train-subgraph.json")
		dev_subg = os.path.join(output_dir, "dev-subgraph.json")
		test_subg = os.path.join(output_dir, "test-subgraph.json")

		entity_dic = os.path.join(output_dir, "entities.txt")
		relation_dic = os.path.join(output_dir, "relations.txt")
		word_dic = os.path.join(output_dir, "vocab.txt")
		entity2id = load_dict(entity_dic)
		word2id = load_dict(word_dic)
		relation2id = load_dict(relation_dic)

		relation_emb_file = os.path.join(output_dir, "relations.npy")
		vocab_emb_file = os.path.join(output_dir, "vocab.npy")
		entity_emb_file = os.path.join(output_dir, "entities.npy")
		train_data, valid_data, test_data = data_load(self.config, train_subg, dev_subg, test_subg, entity2id, word2id, relation2id)

		dev_nerd_file = os.path.join(input_dir, f"dev-nerd.json")
		dev_re_fp = result_path + '/gcn_result_dev.txt'
		dev_re_category_fp = result_path + '/gcn_category_result_dev.txt'
		# result on dev set
		#dev_acc = test(self.config, valid_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file)
		pred_kb_file = os.path.join(os.path.join(self.config['model_folder'], self.config['pred_file']))
		threshold = 0.0
		compare_pr(pred_kb_file, threshold, open(dev_re_fp, 'w', encoding='utf-8'))
		evaluate_result_for_category(dev_nerd_file, dev_re_fp, dev_re_category_fp)
		# result on test set
		test_re_fp = result_path + '/gcn_result_test.txt'
		test_re_category_fp = result_path + '/gcn_category_result_test.txt'
		test_nerd_file = os.path.join(input_dir, f"test-nerd.json")
		test_acc = test(self.config, test_data, entity2id, word2id, entity_emb_file, vocab_emb_file, relation_emb_file)
		pred_kb_file = os.path.join(os.path.join(self.config['model_folder'], self.config['pred_file']))
		compare_pr(pred_kb_file, threshold, open(test_re_fp, 'w', encoding='utf-8'))
		evaluate_result_for_category(test_nerd_file, test_re_fp, test_re_category_fp)

	def _generate_dictionary_embeddings(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = os.path.join(input_dir, self.name, self.nerd)
		train = os.path.join(output_dir, "train-subgraph.json")
		dev = os.path.join(output_dir, "dev-subgraph.json")
		test = os.path.join(output_dir, "test-subgraph.json")
		files = [train, dev, test]

		entity_dic = os.path.join(output_dir, "entities.txt")
		relation_dic = os.path.join(output_dir, "relations.txt")
		word_dic = os.path.join(output_dir, "vocab.txt")
		entity_name_dic = os.path.join(output_dir, "entityname.pkl")
		date_dic = os.path.join(output_dir, "tem_entities.txt")
		trelation_dic = os.path.join(output_dir, "tem_relations.txt")

		#words, entities, relations, trelations, tentities, categories, signals, entityname = get_dictionary(files)

		# fp_entityname = open(entity_name_dic, 'wb')
		# pickle.dump(entityname, fp_entityname)
		#
		# fp_entity = open(entity_dic, 'w', encoding='utf-8')
		# for item in entities:
		# 	fp_entity.write(item)
		# 	fp_entity.write('\n')
		# fp_entity.close()
		#
		# fp_date = open(date_dic, 'w', encoding='utf-8')
		# for item in tentities:
		# 	fp_date.write(item)
		# 	fp_date.write('\n')
		# fp_date.close()
		#
		# fp_rel = open(relation_dic, 'w', encoding='utf-8')
		# for item in relations:
		# 	fp_rel.write(item)
		# 	fp_rel.write('\n')
		# fp_rel.close()
		#
		# fp_trel = open(trelation_dic, 'w', encoding='utf-8')
		# for item in trelations:
		# 	# rel = "|".join(item)
		# 	fp_trel.write(item)
		# 	fp_trel.write('\n')
		# fp_trel.close()
		#
		# fp_word = open(word_dic, 'w', encoding='utf-8')
		# for item in set(words):
		# 	if len(item) > 0:
		# 		fp_word.write(item)
		# 		fp_word.write('\n')
		# print('\n\nlength of words:', str(len(set(words))))
		# if "__unk__" in words:
		# 	print(words.index("__unk__"))
		# fp_word.close()
		#
		# print("#entities = %s" % str(len(entities)))
		# print("#temporal relations = %s" % str(len(trelations)))
		# print("#relations = %s" % str(len(relations)))

		relation_emb_file = os.path.join(output_dir, "relations.npy")
		vocab_emb_file = os.path.join(output_dir, "vocab.npy")
		entity_emb_file = os.path.join(output_dir, "entities.npy")

		# relation encoding, get pretrained_relation_emb_file
		relation_embeddings, relation_embedding_matrix = relation_emb(relation_dic, self.wiki2vec)
		print('Saving Relations....')
		np.save(relation_emb_file, relation_embedding_matrix)
		# question word encoding, get pretrained_word_embedding_file
		print('Embedding Words....')
		vocab_embeddings, vocab_embedding_matrix = word_emb(word_dic, self.wiki2vec)
		print('Saving Vocabs....')
		np.save(vocab_emb_file, vocab_embedding_matrix)
		print('Embedding Entities...')
		entity_embeddings, entity_embedding_matrix = entity_emb(entity_dic, date_dic, entity_name_dic, self.wiki2vec)
		# for ent in no_in_entity:
		#     fo.write(ent)
		#     fo.write('\n')


		print('Saving Entities....')
		np.save(entity_emb_file, entity_embedding_matrix)

	def _nerd_entities(self, input_path):
		with open(input_path, "r") as fp:
			data = json.load(fp)
		self.logger.info(f"Input data loaded from: {input_path}.")

		wiki_ids = list()
		for instance in tqdm(data):
			wiki_ids += instance["elq"]
			if self.nerd == "elq-wat":
				wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
			elif self.nerd == "elq-tagme":
				wiki_ids += instance["tagme"]
			elif self.nerd == "elq-tagme-wat":
				wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
				wiki_ids += instance["tagme"]

		return wiki_ids

	# def _twohop_facts_retriever(self):
	# 	"""
	# 	Run the pipeline using gold answers for the dataset.
	# 	"""
	# 	# define output path
	# 	"""Run ERS on data and add retrieve top-e evidences for each source combination."""
	# 	input_dir = self.config["path_to_intermediate_results"]
	# 	output_dir = self.config["path_to_intermediate_results"]
	# 	output_path = os.path.join(output_dir, self.name, self.nerd)
	# 	Path(output_path).mkdir(parents=True, exist_ok=True)
	#
	# 	output_path = os.path.join(output_path, f"nerd-two-hop-facts.pkl")
	# 	# process data
	# 	input_path_test = os.path.join(input_dir, f"test-nerd.json")
	# 	input_path_dev = os.path.join(input_dir, f"dev-nerd.json")
	# 	input_path_train = os.path.join(input_dir, f"train-nerd.json")
	#
	# 	nerd_entities = self._nerd_entities(input_path_test)
	# 	nerd_entities += self._nerd_entities(input_path_dev)
	# 	nerd_entities += self._nerd_entities(input_path_train)
	#
	# 	two_hop_facts = self.ftr.twohop_fact_retriever(nerd_entities)
	#
	# 	pickle.dump(two_hop_facts, open(output_path, 'wb'))

	def _evaluate_retriever_dev(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		output_path = os.path.join(output_dir, self.name, self.nerd)
		Path(output_path).mkdir(parents=True, exist_ok=True)

		input_path = os.path.join(input_dir, f"dev-nerd.json")
		output_path = os.path.join(output_path, f"dev-towhop-er.jsonl")
		self.er_inference_on_data_split(input_path, output_path)
		self.evaluate_retrieval_results(output_path)

	def _evaluate_retriever_train(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		output_path = os.path.join(output_dir, self.name, self.nerd)
		Path(output_path).mkdir(parents=True, exist_ok=True)

		input_path = os.path.join(input_dir, f"train-nerd.json")
		output_path = os.path.join(output_path, f"train-towhop-er.jsonl")
		self.er_inference_on_data_split(input_path, output_path)
		self.evaluate_retrieval_results(output_path)

	def er_inference_on_data_split(self, input_path, output_path):
		"""
        Run Two Hop Fact Retrieval on the dataset.
        """
		# open data
		with open(input_path, "r") as fp:
			data = json.load(fp)
		self.logger.info(f"Input data loaded from: {input_path}.")

		# process data
		with open(output_path, "w") as fp:
			for instance in tqdm(data):
				evidences = self.er_inference_on_instance(instance)
				instance["candidate_evidences"] = evidences
				instance["answers"] = format_answers(instance)
				# answer presence
				hit, answering_evidences = answer_presence(evidences, instance["answers"])
				instance["answer_presence"] = hit

				# write instance to file
				fp.write(json.dumps(instance))
				fp.write("\n")

		# log
		self.logger.info(f"Evaluating retrieval results: {output_path}.")
		#self.evaluate_retrieval_results(output_path)
		self.logger.info(f"Done with processing: {input_path}.")

	def er_inference_on_instance(self, instance):
		wiki_ids = list()
		wiki_ids = instance["elq"]
		if self.nerd == "elq-wat":
			wiki_ids += [[item[0][0], item[1], item[2]] for item in instance["wat"]]
		elif self.nerd == "elq-tagme":
			wiki_ids += instance["tagme"]
		elif self.nerd == "elq-tagme-wat":
			wiki_ids += instance["wat"]
			wiki_ids += instance["tagme"]
		evidences = self.ftr.twohop_fact_retriever_previous_version(wiki_ids)
		return evidences

	def evaluate_retrieval_results(self, results_path):
		"""
        Evaluate the results of the retrieval phase, for
        each source, and for each category.
        """
		# score
		answer_presences = list()
		category_to_ans_pres = {"explicit": [], "implicit": [], "ordinal": [], "temp.ans": [], "all": []}
		category_to_evi_num = {"explicit": [], "implicit": [], "ordinal": [], "temp.ans": [], "all": []}

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
		res_path = results_path.replace(".jsonl", "-retrieval.res")
		with open(res_path, "w") as fp:
			fp.write(f"evaluation result:\n")
			category_answer_presence_per_src = {
				category: (sum(num) / len(num)) for category, num in category_to_ans_pres.items() if len(num) != 0
			}
			fp.write(f"Category Answer presence per source: {category_answer_presence_per_src}")

			for category in category_to_evi_num:
				fp.write("\n")
				fp.write(f"category: {category}\n")
				fp.write(
					f"Avg. evidence number: {sum(category_to_evi_num[category]) / len(category_to_evi_num[category])}\n")
				sorted_category_num = category_to_evi_num[category]
				sorted_category_num.sort()
				fp.write(f"Max. evidence number: {sorted_category_num[-1]}\n")
				fp.write(f"Min. evidence number: {sorted_category_num[0]}\n")

	def _load_property(self):
		self.logger.info("Loading Property Dictionary")
		return get_property(self.property_path)

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise Exception("Usage: python exaqt/graft/pipeline.py <FUNCTION> <PATH_TO_CONFIG>")

	# load config
	function = sys.argv[1]
	config_path = sys.argv[2]
	config = get_config(config_path)

	# inference using predicted answers
	# inference using predicted answers
	if function == "--retrieve-twohop":
		pipeline = Pipeline(config)
		pipeline._twohop_facts_retriever()

	elif function == "--retrieve-dev":
		pipeline = Pipeline(config)
		pipeline._evaluate_retriever_dev()

	elif function == "--retrieve-train":
		pipeline = Pipeline(config)
		pipeline._evaluate_retriever_train()

	elif function == "--subgraph-dev":
		pipeline = Pipeline(config)
		pipeline._generate_subgraph_dev()

	elif function == "--subgraph-train":
		pipeline = Pipeline(config)
		pipeline._generate_subgraph_train()

	elif function == "--subgraph-test":
		pipeline = Pipeline(config)
		pipeline._generate_subgraph_test()

	elif function == "--embedding":
		pipeline = Pipeline(config)
		pipeline._relation_question_embeddings()

	elif function == "--dictionary":
		pipeline = Pipeline(config)
		pipeline._generate_dictionary_embeddings()

	elif function == "--train":
		pipeline = Pipeline(config)
		pipeline.gcn_model_train()

	elif function == "--inference":
		pipeline = Pipeline(config)
		pipeline.gcn_model_inference()

	else:
		raise Exception(f"Unknown function {function}!")
