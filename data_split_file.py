#!/usr/bin/env python

import pandas as pd
import numpy as np
from glob import glob
import os

# paths to the data files
audio_path = os.path.join(os.getcwd(),"BembaSpeech/bem/audio/")
csv_path = os.path.join(os.getcwd(),"BembaSpeech/bem/splits/")

def create_gender_partitions(csv_path):
    csv_file_list = glob(f"{csv_path}/*.csv")
    speaker_file_splits=["Male","Female"]
    for gender_split in speaker_file_splits:
        for csv_file in csv_file_list:
            split_file = os.path.basename(csv_file).split(".")[0]
            df = pd.read_csv(csv_file, sep="\t")
            df = df[df["speaker_gender"]==gender_split]
            df.to_csv(f"{csv_path}/{split_file}_{gender_split}.csv", sep="\t", index=False)
            print(f"{split_file}_{gender_split}.csv - Number of Records:", len(df))

def prepare_data(audio_path, csv_path):
    csv_file_list = glob(f"{csv_path}/*.csv")
    for csv_file in csv_file_list:
        split_file = os.path.basename(csv_file).split(".")[0]
        df = pd.read_csv(csv_file, sep="\t")
        df["path"] = audio_path + df['audio']
        df = df.dropna(subset=["path"])
        df = df.drop(columns=['audio'])
        df = df.rename(columns={'path':'audio'})
        df = df[["audio","sentence"]]
        df.to_csv(f"{csv_path}/{split_file}_processed.tsv", sep="\t", index=False)
        print(f"{split_file}_processed : ", len(df))

if __name__== "__main__":
    # run the python function to prepare the data
    create_gender_partitions(csv_path)
    prepare_data(audio_path, csv_path)