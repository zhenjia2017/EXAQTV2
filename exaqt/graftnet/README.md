Evaluating GraftNET KGQA part on TimeQuestions or other benchmarks
============

## Environment setup
You need to run the program under the environment of EXAQT.

### Dependencies
We make use of [CLOCQ](https://github.com/PhilippChr/CLOCQ) for retrieving two-hop facts.
CLOCQ can be conveniently integrated via the [publicly available API](https://clocq.mpi-inf.mpg.de), using the client from [the repo](https://github.com/PhilippChr/CLOCQ).  

We make use of [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq) and [TagMe](https://sobigdata.d4science.org/web/tagme/tagme-help) (or [WAT](https://sobigdata.d4science.org/web/tagme/wat-api)) to run NERD which is for comparison with EXAQT.
You need to install and build the environment for running [ELQ](https://github.com/facebookresearch/BLINK/tree/master/elq). We already integrated [TagMe](https://sobigdata.d4science.org/web/tagme/tagme-help) (or [WAT](https://sobigdata.d4science.org/web/tagme/wat-api)) in the scripts for NERD.

### Data
You need the pretrained model:
- Wikipedia2Vec model. You can download from [here](https://qa.mpi-inf.mpg.de/exaqt/exaqt-supp-data.zip).

## Evaluating GraftNET KGQA on TimeQuestions
For reproducing the GraftNET KGQA results on TimeQuestions, run the following script:
``` bash
    bash scripts/pipeline-grafnet.sh --graftnet-pipeline config/timequestions/config.yml
```

You also can use the following command to start the program. 
``` command
    python -u exaqt/graftnet/pipeline.py --graftnet-pipeline config/timequestions/config.yml
```

## Evaluating GraftNET KGQA on other benchmarks

- Reformat the benchmark as the same format as the TimeQuestions
- Put the reformatted benchmark under the "[_benchmarks](_benchmarks)" folder 
- Update the config.yml with replacing the name of "benchmark", "train_input_path", "dev_input_path" and "test_input_path" respectively


We provide an example benchmark named "dataset_for_test_pipeline". For evaluating this benchmark, run the following script:
``` bash
    bash scripts/pipeline-grafnet.sh --graftnet-pipeline config/dataset_for_test_pipeline/config.yml
```

You also can use the following command to start the program.
``` command
    python -u exaqt/graftnet/pipeline.py --graftnet-pipeline config/dataset_for_test_pipeline/config.yml
```




