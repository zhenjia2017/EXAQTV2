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

## Reproduce paper results
First, you need to run the scripts for NERD on the dataset. Please follow the instruction in README-exaqt-nerd.md to run NERD on the dataset.

EXAQT includes two stages: answer graph construction and answer prediction.
For reproducing the results on TimeQuestions, run the following two commands respectively:
``` bash
    bash scripts/pipeline.sh --answer-graph-pipeline config/timequestions/config.yml
    bash scripts/pipeline.sh --answer-predict config/timequestions/config.yml
```


