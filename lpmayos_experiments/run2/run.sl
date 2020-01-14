#!/bin/bash
#SBATCH --job-name="squad2"
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=20Gb
#SBATCH -p high
#SBATCH --gres=gpu:1

module load Python/3.6.4-foss-2017a
module load cuDNN/7.6.3.30-CUDA-10.0.130

conda activate env36

SQUAD_DIR=../../examples/tests_samples/SQUAD
OUTPUT_DIR=./results

python ../../examples/run_squad.py --model_type bert \
                                --model_name_or_path bert-base-cased \
                                --do_train \
                                --do_eval \
                                --train_file $SQUAD_DIR/train-v2.0.json \
                                --predict_file $SQUAD_DIR/dev-v2.0.json \
                                --learning_rate 3e-5 \
                                --num_train_epochs 20 \
                                --max_seq_length 384 \
                                --doc_stride 128 \
                                --output_dir $OUTPUT_DIR/ \
                                --per_gpu_eval_batch_size=3   \
                                --per_gpu_train_batch_size=3   \
