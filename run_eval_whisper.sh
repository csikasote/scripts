#!/bin/bash

#model_id=(
#  "openai/whisper-medium" \
#  "csikasote/whisper-medium-bemgen-male-model" \
#  "csikasote/whisper-medium-bemgen-female-model" \
#  "csikasote/whisper-medium-bemgen-balanced-model" \
#  "csikasote/whisper-medium-bemgen-combined-model")

model_id="csikasote/whisper-medium-bemgen-male-model"
dataset="bemgen"
split_list=("male" "female" "combined")
file_name="combined"

for split_name in ${split_list[@]};
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