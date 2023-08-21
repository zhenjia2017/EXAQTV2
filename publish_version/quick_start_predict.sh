#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 2-00:00:00
#SBATCH -o prediction_0404.log
#SBATCH --gres gpu:1


# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
python3 step1-2_question_xg_spofact_score.py  --use-gpu ${CUDA_VISIBLE_DEVICES}