#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 2-00:00:00
#SBATCH -o train_model_phase1_onehop_and.log
#SBATCH --gres gpu:1


# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
python3 step4_relevant_fact_selection_model.py -t 1