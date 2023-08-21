import os
import sys
import logging

from exaqt.library.utils import get_config, get_logger, get_result_logger, store_json_with_mkdir, get_property
from exaqt.answer_graph.ftrs import FTRS

class Pipeline:
	def __init__(self, config):
		"""Create the pipeline based on the config."""
		# load config
		self.config = config
		self.logger = get_logger(__name__, config)
		self.result_logger = get_result_logger(config)
		self.property_path = self.config["pro-info-path"]
		self.property = self._load_property()
		self.ftrs = self._load_ftrs()
		# load individual modules
		self.name = self.config["name"]
		self.nerd = self.config["nerd"]
		loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
		print("Loggers", loggers)

	def run_ftrs(self):
		self.ftrs.train()
		self.ftrs.ers_inference()

	def answer_graph(self):
		self._evaluate_retriever_train()
		self._evaluate_retriever_test()
		self._evaluate_retriever_dev()
		self.run_ftrs()
		self.run_ers_inference_test()

	def run_ers_inference_test(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"test-er.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}.jsonl")

		self.ftrs.ers_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def run_ers_inference_dev(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"dev-er.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")

		self.ftrs.ers_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def run_ers_inference_train(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"train-er.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}.jsonl")

		self.ftrs.ers_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def run_tempftrs(self):
		self.ftrs.temporal_train()
		self.ftrs.temporal_ers_inference()

	def _evaluate_seed_path_test(self):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		"""Run ERS on data and add retrieve top-e evidences for each source combination."""
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}.jsonl")
		# process data
		output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-path.jsonl")
		self.ftrs.path_inference_on_data_split(input_path, output_path)

	def _evaluate_seed_path_dev(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path.jsonl")
		self.ftrs.path_inference_on_data_split(input_path, output_path)

	def _evaluate_seed_path_train(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
		self.ftrs.path_inference_on_data_split(input_path, output_path)

	def _evaluate_gst_test(self):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		"""Run GST on data"""
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"test-ers-{fs_max_evidences}-path.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"test-ers-{fs_max_evidences}-gst.jsonl")

		self.ftrs.gst_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_gst_results(output_path)

	def _evaluate_gst_dev(self):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		"""Run GST on data"""
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"dev-ers-{fs_max_evidences}-path.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"dev-ers-{fs_max_evidences}-gst.jsonl")

		self.ftrs.gst_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_gst_results(output_path)

	def _evaluate_gst_train(self):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		"""Run GST on data"""
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		fs_max_evidences = self.config["fs_max_evidences"]
		# process data
		input_path = os.path.join(input_dir, self.nerd, f"train-ers-{fs_max_evidences}-path.jsonl")
		output_path = os.path.join(output_dir, self.nerd, f"train-ers-{fs_max_evidences}-gst.jsonl")

		self.ftrs.gst_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_gst_results(output_path)

	def _evaluate_retriever_test(self):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		"""Run ERS on data and add retrieve top-e evidences for each source combination."""
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]

		# process data
		input_path = os.path.join(input_dir, f"test-nerd.json")
		output_path = os.path.join(output_dir, self.nerd, f"test-er.jsonl")
		self.ftrs.er_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def _evaluate_retriever_dev(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		input_path = os.path.join(input_dir, f"dev-nerd.json")
		output_path = os.path.join(output_dir, self.nerd, f"dev-er.jsonl")
		self.ftrs.er_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def _evaluate_retriever_train(self):
		input_dir = self.config["path_to_intermediate_results"]
		output_dir = self.config["path_to_intermediate_results"]
		input_path = os.path.join(input_dir, f"train-nerd.json")
		output_path = os.path.join(output_dir, self.nerd, f"train-er.jsonl")
		self.ftrs.er_inference_on_data_split(input_path, output_path)
		self.ftrs.evaluate_retrieval_results(output_path)

	def _load_ftrs(self):
		self.logger.info("Loading Fact Retrieval module")
		return FTRS(self.config, self.property)

	def _load_property(self):
		self.logger.info("Loading Property Dictionary")
		return get_property(self.property_path)

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise Exception("Usage: python exaqt/anp-pipeline.py <FUNCTION> <PATH_TO_CONFIG>")

	# load config
	function = sys.argv[1]
	config_path = sys.argv[2]
	config = get_config(config_path)

	# inference using predicted answers
	if function == "--path-test":
		pipeline = Pipeline(config)
		pipeline._evaluate_seed_path_test()

	elif function == "--path-dev":
		pipeline = Pipeline(config)
		pipeline._evaluate_seed_path_dev()

	elif function == "--path-train":
		pipeline = Pipeline(config)
		pipeline._evaluate_seed_path_train()

	# inference using predicted answers
	elif function == "--retrieve-test":
		pipeline = Pipeline(config)
		pipeline._evaluate_retriever_test()

	elif function == "--retrieve-dev":
		pipeline = Pipeline(config)
		pipeline._evaluate_retriever_dev()

	elif function == "--retrieve-train":
		pipeline = Pipeline(config)
		pipeline._evaluate_retriever_train()

	elif function == "--run_ftrs":
		pipeline = Pipeline(config)
		pipeline.run_ftrs()

	elif function == "--run_tempftrs":
		pipeline = Pipeline(config)
		pipeline.run_tempftrs()

	elif function == "--gst-test":
		pipeline = Pipeline(config)
		pipeline._evaluate_gst_test()

	elif function == "--gst-dev":
		pipeline = Pipeline(config)
		pipeline._evaluate_gst_dev()

	elif function == "--gst-train":
		pipeline = Pipeline(config)
		pipeline._evaluate_gst_train()

	elif function == "--fts-test":
		pipeline = Pipeline(config)
		pipeline.run_ers_inference_test()

	elif function == "--fts-dev":
		pipeline = Pipeline(config)
		pipeline.run_ers_inference_dev()

	elif function == "--fts-train":
		pipeline = Pipeline(config)
		pipeline.run_ers_inference_train()

	else:
		raise Exception(f"Unknown function {function}!")
