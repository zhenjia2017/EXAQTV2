import os
import sys
import logging
from pathlib import Path

from library.utils import get_config, get_logger, get_result_logger, store_json_with_mkdir
from get_seed_entity import EntityLinkELQWATMatch

class NerdPipeline:
	def __init__(self, config):
		"""Create the pipeline based on the config."""
		# load config
		self.config = config
		self.logger = get_logger(__name__, config)
		self.result_logger = get_result_logger(config)
		self.nerd = self._load_nerd()
		# load individual modules
		self.name = self.config["name"]
		loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
		print("Loggers", loggers)

	def run_nerd(self, split):
		"""
		Run the pipeline using gold answers for the dataset.
		"""
		# define output path
		benchmark = self.config["benchmark"]
		benchmark_path = self.config["benchmark_path"]
		output_dir = os.path.join(self.config["path_to_intermediate_results"], benchmark)
		Path(output_dir).mkdir(parents=True, exist_ok=True)

		# process data
		if split == "test":
			input_path = os.path.join(benchmark_path, benchmark, config["test_input_path"])
			output_path = os.path.join(output_dir, f"test-nerd.json")
			self.nerd.inference_on_data_split(input_path, output_path)
		elif split == "dev":
			input_path = os.path.join(benchmark_path, benchmark, config["dev_input_path"])
			output_path = os.path.join(output_dir, f"dev-nerd.json")
			self.nerd.inference_on_data_split(input_path, output_path)
		elif split == "train":
			input_path = os.path.join(benchmark_path, benchmark, config["train_input_path"])
			output_path = os.path.join(output_dir, f"train-nerd.json")
			self.nerd.inference_on_data_split(input_path, output_path)
		# store results in cache
		self.nerd.store_cache()

	def _load_nerd(self):
		self.logger.info("Loading NERD module")
		return EntityLinkELQWATMatch(self.config)

	def example(self):
		"""Run pipeline on a single input instance."""
		input_data = [
				{
					"Id": 12945,
					"Question": "what time is justin biebers birthday",
					"Temporal signal": [
						"No signal"
					],
					"Temporal question type": [
						"Temp.Ans"
					],
					"Answer": [
						{
							"AnswerType": "Value",
							"AnswerArgument": "1994-03-01T00:00:00Z"
						}
					],
					"Data source": "ComQA (NAACL 2019)",
					"Question creation date": "2019-06-03",
					"Data set": "test"
				}
        ]

		# run inference
		self.nerd.inference_on_data(input_data)
		# store processed data
		output_dir = self.config["path_to_intermediate_results"]
		ner = self.config["nerd"]
		output_dir = os.path.join(output_dir, "test", ner)
		Path(output_dir).mkdir(parents=True, exist_ok=True)
		output_path = os.path.join(output_dir, f"{ner}.json")
		store_json_with_mkdir(input_data, output_path)
		self.logger.info(f"result for the example: {input_data}")


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise Exception("Usage: python pipeline_nerd.py <FUNCTION> <PATH_TO_CONFIG>")

	# load config
	function = sys.argv[1]
	config_path = sys.argv[2]
	config = get_config(config_path)
	pipeline = NerdPipeline(config)

	# inference using predicted answers
	if function == "--example":
		pipeline.example()

	elif function == "--nerd-all":
		pipeline.run_nerd("train")
		pipeline.run_nerd("dev")
		pipeline.run_nerd("test")

	elif function == "--nerd-test":
		pipeline.run_nerd("test")

	elif function == "--nerd-dev":
		pipeline.run_nerd("dev")

	elif function == "--nerd-train":
		pipeline.run_nerd("train")

	else:
		raise Exception(f"Unknown function {function}!")
