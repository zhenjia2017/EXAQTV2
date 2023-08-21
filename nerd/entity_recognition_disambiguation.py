import json

from tqdm import tqdm

from library.utils import store_json_with_mkdir, get_logger

class NERD:
    def __init__(self, config):
        """Initialize nerd module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def inference(self, input_path, output_path):
        """Run baseline model on data and add answers."""
        self.inference_on_data_split(input_path, output_path)

    def inference_on_instance(self, instance):
        raise Exception("This is an abstract function which should be overwritten in a derived class!")

    def inference_on_data_split(self, input_path, output_path):
        """
        Answer the given dataset and store the output in the specified path.
        """
        with open(input_path, "r") as fp:
            input_data = json.load(fp)
            self.inference_on_data(input_data)
            # store processed data
            store_json_with_mkdir(input_data, output_path)

    def inference_on_data(self, input_data):
        """Run NERD on the given data."""
        for instance in tqdm(input_data):
            self.inference_on_instance(instance)


