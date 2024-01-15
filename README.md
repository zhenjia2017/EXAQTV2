EXAQTV2
============

This repository is for the updated code version of EXAQT. It helps to reproduce the results of the [paper](https://arxiv.org/abs/2109.08935) or also can be used for evaluating other KGQA benchmarks.

## Environment setup
We recommend the installation via conda, and provide the corresponding environment file in [environment.yml](environment.yml):

```bash
    git clone https://github.com/zhenjia2017/EXAQTV2.git
    cd EXAQTV2/
    conda env create --file environment.yml
    conda activate exaqt
    pip install -e .
```
Alternatively, you can also install the requirements via pip, using the [requirements.txt](requirements.txt) file. 

### Dependencies
EXAQT makes use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving facts for constructing answer graph.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client from [the repo](https://github.com/PhilippChr/CLOCQ).  

EXAQT makes use of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) and [TagMe](https://sobigdata.d4science.org/web/tagme/tagme-help) (or [WAT](https://sobigdata.d4science.org/web/tagme/wat-api)) to run NERD.
You need to install and build the environment for running [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq). We already integrated [TagMe](https://sobigdata.d4science.org/web/tagme/tagme-help) (or [WAT](https://sobigdata.d4science.org/web/tagme/wat-api)) in the scripts for NERD.

### Data
You need two pretrained models:
- BERT model. You can download from [here](https://huggingface.co/bert-base-cased/tree/main).
- Wikipedia2Vec model. You can download from [here](https://qa.mpi-inf.mpg.de/exaqt/exaqt-supp-data.zip).

## Reproduce paper results
First, you need to run the scripts for NERD on the dataset. Please follow the instruction in README-exaqt-nerd.md to run NERD on the dataset.
Then you need to run the script pipeline.py. EXAQT includes two stages: answer graph construction and answer prediction.
For reproducing the results on TimeQuestions, run the following two scripts respectively:
``` bash
    bash scripts/pipeline.sh --answer-graph config/timequestions/config.yml
    bash scripts/pipeline.sh --answer-predict config/timequestions/config.yml
```

You also can use the following commands to start the pipeline of answer graph construction and answer prediction.
``` command
    python -u exaqt/pipeline.py --answer-graph config/timequestions/config.yml
    python -u exaqt/pipeline.py --answer-predict config/timequestions/config.yml
```

## Evaluating EXAQT on other benchmarks

- Reformat the benchmark as the same format as the TimeQuestions
- Put the reformatted benchmark under the "[_benchmarks](_benchmarks)" folder 
- Update the config.yml with replacing the name of "benchmark", "train_input_path", "dev_input_path" and "test_input_path" respectively

We provide an example benchmark named "dataset_for_test_pipeline". For evaluating this benchmark, run the following two scripts respectively:

``` bash
    bash scripts/pipeline.sh --answer-graph config/dataset_for_test_pipeline/config.yml
    bash scripts/pipeline.sh --answer-predict config/dataset_for_test_pipeline/config.yml
```

You also can use the following commands to start the pipeline of answer graph construction and answer prediction.
``` command
    python -u exaqt/pipeline.py --answer-graph config/dataset_for_test_pipeline/config.yml
    python -u exaqt/pipeline.py --answer-predict config/dataset_for_test_pipeline/config.yml
```



