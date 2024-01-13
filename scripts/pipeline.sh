#!/usr/bin/bash
#SBATCH -o out/slurm/out

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify at least the pipeline-function."
	echo "Usage: bash scripts/pipeline.sh\\
		--answer-graph\\
		/--answer-prediction\\
		[<PATH_TO_CONFIG>]"
	exit 0
fi

## read config parameter: if no present, stick to default (exaqt-elq-wat.yml)
FUNCTION=$1
CONFIG=${2:-"config/timequestions/config.yml"}

# set path for output
# get function name
FUNCTION_NAME=${FUNCTION#"--"}
# get data name
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
# get config name
CFG_NAME=${NAME[2]%".yml"}

 #set output path (include sources only if not default value)
if [[ $# -lt 3 ]]
then
	OUT="out/${DATA}/pipeline-${FUNCTION_NAME}-${CFG_NAME}.out"
fi

echo $OUT


## start script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	nohup python -u exaqt/pipeline.py $FUNCTION $CONFIG $OUT 2>&1 &
else
	echo "starting slurm task now"
	echo $OUT
	# run with sbatch
	sbatch <<EOT
#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -t 2-00:00:00

python -u exaqt/pipeline.py $FUNCTION $CONFIG
EOT
fi
