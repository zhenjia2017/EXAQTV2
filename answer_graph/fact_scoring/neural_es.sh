#!/usr/bin/bash
#SBATCH -o neural_es/out/fail.out

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify at least the pipeline-function."
	echo "Usage: bash neural_es/neural_es.sh\\
		--train
		[<PATH_TO_CONFIG>]"
	exit 0
fi

## read config parameter: if no present, stick to default (default.yaml)
FUNCTION=$1
CONFIG=${2:-"EXAQT/config/timequestionsv2/exaqt-elq-wat.yml"}

## set path for output
# get function name
FUNCTION_NAME=${FUNCTION#"--"}
# get data name
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
# get config name
CFG_NAME=${NAME[2]%".yml"}
OUT="neural_es/out/${FUNCTION_NAME}-${CFG_NAME}.out"

echo $OUT

## start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup

	nohup python -u neural_es $FUNCTION $CONFIG > $OUT 2>&1 &
else
	# run with sbatch
	sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$OUT
#SBATCH -p gpu20
#SBATCH --gres gpu:4
#SBATCH -t 15:00:00
#SBATCH -o $OUT
#SBATCH -d singleton

python -u neural_es $FUNCTION $CONFIG 
EOT
fi
