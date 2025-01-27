#!/bin/bash

model_ids=(
  "csikasote/whisper-medium-bigcgen-male-5hrs-model" \
  "csikasote/whisper-medium-bigcgen-female-5hrs-model" \
  "csikasote/whisper-medium-bigcgen-balanced-model" \
  "csikasote/whisper-medium-bigcgen-combined-5hrs-model" \
  "csikasote/whisper-medium-bigcgen-combined-10hrs-model" \
  "csikasote/whisper-medium-bigcgen-combined-15hrs-model" \
  "csikasote/whisper-medium-bigcgen-combined-20hrs-model" \
  "csikasote/whisper-medium-bigcgen-combined-25hrs-model" \
  "csikasote/whisper-medium-bigcgen-combined-30hrs-model")

#model_id="openai/whisper-medium"
dataset="bigcgen"
split_list=("male" "female" "combined")
split_name="test"

for model_id in ${model_ids[@]};
do
  echo $model_id
  for file_name in ${split_list[@]};
  do
    csv_test_path="/content/${dataset}/splits/${split_name}/test_${file_name}_file_processed.tsv"
    output_file="/content/${split_name}_wer.csv"
    echo $split_name
    python run_eval_whisper_modelv2.py \
      --model_id=$model_id \
      --dataset=$dataset \
      --config="en" \
      --streaming="False" \
      --path=$csv_test_path \
      --output=$output_file \
      --device=0 \
      --language="en" \
      --batch_size=8 \
      --task="transcribe" 
  done
  python compute_ANOVA.py
done 