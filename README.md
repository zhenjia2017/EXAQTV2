EXAQTV2
============

This repository is for the updated code version of EXAQT. It helps to reproduce the results of the [paper](https://arxiv.org/abs/2109.08935) or is used for other kinds of benchmarks.

## Environment setup
Clone the repo via:
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

[ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) is one of NERD tools used by EXAQT. So you need to install and build the environment for running ELQ.

## Reproduce paper results
First, you need to process NERD on the dataset. You need to put the scripts in the "nerd" folder under the "BLINK" folder after installing "ELQ".
Please follow the instruction in README-exaqt-nerd.md to run NERD on the dataset.

EXAQT includes two stages: answer graph construction and answer prediction.
For reproducing the results on TimeQuestions, run the following two commands respectively:
``` bash
    bash scripts/pipeline.sh --answer-graph-pipeline config/timequestions/config.yml
    bash scripts/pipeline.sh --answer-predict config/timequestions/config.yml
```


