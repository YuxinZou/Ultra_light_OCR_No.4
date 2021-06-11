#! /usr/bin/env bash

set -o pipefail

TASK=$1
DATASET=$2
GPU=$3

CUDA_VISIBLE_DEVICES=${GPU} python tools/eval.py -c configs/rec/new_baseline/${TASK}.yml -o Eval.dataset.label_file_list=["./data/train_data/${DATASET}.txt"] Global.pretrained_model="./checkpoint/${TASK}/best_accuracy"
