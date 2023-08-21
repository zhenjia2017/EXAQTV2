#!/usr/bin/bash
#SBATCH -o out/slurm/out

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify at least the pipeline-function."
	echo "Usage: bash scripts/pipeline.sh\\
		--train\\
		/--pred-answers\\
		/--gold-answers\\
		/--main-results\\
		/--ans-pres\\
		/--example\\
		/--train-[qu/ers/ha]\\
		[<PATH_TO_CONFIG>] [<SOURCES_STR>]"
	exit 0
fi

## read config parameter: if no present, stick to default (default.yaml)
FUNCTION=$1
CONFIG=${2:-"config/timequestionsv2/nerd.yml"}

## set path for output
# get function name
FUNCTION_NAME=${FUNCTION#"--"}
# get data name
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
# get config name
CFG_NAME=${NAME[2]%".yml"}

# set output path (include sources only if not default value)
if [[ $# -lt 3 ]]
then
	OUT="out/${DATA}/pipeline-${FUNCTION_NAME}-${CFG_NAME}.out"
else
	OUT="out/${DATA}/pipeline-${FUNCTION_NAME}-${CFG_NAME}.out"
fi

echo $OUT

## fix global vars (required for FiD)
export SLURM_NTASKS=1
export TOKENIZERS_PARALLELISM=false

## start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	nohup python -u exaqt/answer_graph/connectivity/seed_path_extractor.py $FUNCTION $CONFIG > $OUT 2>&1 &
else
	# run with sbatch
	sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o $OUT
#SBATCH -d singleton

python -u exaqt/answer_graph/connectivity/seed_path_extractor.py $FUNCTION $CONFIG
EOT
fi
