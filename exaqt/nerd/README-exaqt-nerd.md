# NERD (Named Entity Recognition and Disambiguation)

Module to recognize entities and link to Wikidata QID for the questions.

- [Create your own NERD module](#create-your-own-nerd-module)
  - [`inference_on_instance` function](#inference_on_instance function)
  - [Run nerd pipeline for your own datas](#run-nerd-pipeline-for-your-own-datas)
  - [`run_nerd` function](#run-nerd-function)

## Create your own NERD module
You can inherit from the [NERD](entity_recognition_disambiguation.py) class and create your own NERD module. Implementing the functions `inference_on_instance` is sufficient for the pipeline to run properly.

Further, you need to instantiate a logger in the class, which will be used in the parent class.
Alternatively, you can call the __init__ method of the parent class.

## `inference_on_instance` function

**Inputs**:
- `instance`: a json object in which there is a key called "Question".

**Description**:  
This method is supposed to generate the nerd results from ELQ, tagme and wat for the current question.

**Output**:  
Returns the instance and store the nerd results in "tagme", "elq":wiki_ids_elq, and "wat" respectively. 

## Run nerd pipeline for your own datas
You can initiate the [NerdPipeline](pipeline_nerd.py) class and process NERD on your own dataset. 

## `run_nerd` function

**Usage**:
python `pipeline_nerd.py` `<FUNCTION>` `<PATH_TO_CONFIG>`

Note that the program should run under directory of BLINK
after building the environment of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq).
You need to put the scripts under the "nerd" folder into the "BLINK" folder after installing "ELQ".

If you only use TagMe or WAT as the NERD tool, you can remove ELQ part from the code.

Other script needed for running the program is utils.py under the library directory.
