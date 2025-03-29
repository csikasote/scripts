"""
Downloads and creates manifest files for speech recognition with Zambezi Voice datasets.

Authors:
 * Claytone Sikasote, 2025
"""

import os
import csv
import json
import shutil
import librosa
import numpy as np
import pandas as pd
from git import Repo
from glob import glob

from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
GITHUB_REPO_URL = "https://github.com/csikasote/nyagen.git"
SAMPLERATE = 16000

def prepare_mini_librispeech(
    data_folder, save_json_train, save_json_valid, save_json_test
):
    """
    Prepares the json files for the Mini Librispeech dataset.

    Downloads the dataset if its not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the Mini Librispeech dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = '/path/to/mini_librispeech'
    >>> prepare_mini_librispeech(data_folder, 'train.json', 'valid.json', 'test.json')
    """
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    destination = os.path.join(data_folder, "nyagen")
    if not check_folders(destination):
        #download_mini_librispeech(data_folder)
        repo_url = GITHUB_REPO_URL
        clone_github_repo(repo_url, destination)

    # Remove processed files
    splits_path = os.path.join(destination, "splits")
    remove_processed_files(splits_path)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    audio_path = os.path.join(destination, "audio")
    prepare_dataset(audio_path, splits_path)

    test_split_path = os.path.join(splits_path, "combined","test_processed.tsv")
    valid_split_path = os.path.join(splits_path, "combined","valid_processed.tsv")
    train_split_path = os.path.join(splits_path, "combined","train_processed.tsv")
    tsv_to_json(test_split_path, save_json_test)
    tsv_to_json(valid_split_path, save_json_valid)
    tsv_to_json(train_split_path, save_json_train)

def prepare_dataset(audio_path, csv_path):
    split_list = ["combined"]
    for split in split_list:
        csv_file_list = glob(f"{csv_path}/{split}/*.tsv")
        #print(csv_file_list)
        for csv_file in csv_file_list:
          split_file = os.path.basename(csv_file).split(".")[0]
          split_file = split_file.split("_")[0]
          df = pd.read_csv(csv_file, sep="\t")
          df["wav"] = audio_path + "/" + df['audio']
          df["length"] = df["wav"].apply(lambda x: get_audio_duration(x))
          df["id"] = df["audio"].apply(lambda x: x[:-4])
          df["domain"] = df["speaker_gender"].apply(lambda x: 1 if x == "Male" else 0)
          df = df.dropna(subset=["wav"])
          df = df.drop(columns=['audio'])
          df = df.rename(columns={'sentence':'words'})
          df = df[["id","wav","words", "length", "domain"]]
          df.to_csv(f"{csv_path}/{split}/{split_file}_processed.tsv", sep="\t", index=False)
          print(f"{split_file}_processed : ", len(df))
          tsv_file = f"{csv_path}/{split}/{split_file}_processed.tsv"
          logger.info(f"Created file: {tsv_file}")
          txt_file = f"../LM/data/{split_file}.txt"
          df_column_to_txt(df, txt_file)

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames: tuple
        The path to files that should exist in order to consider
        preparation already completed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def clone_github_repo(repo_url, destination):
  print(f"Cloning repository: {repo_url}")
  try:
      Repo.clone_from(repo_url, destination)
      print(f"Repository cloned successfully to {destination}")
  except Exception as e:
      print(f"Error cloning repository: {e}")

def remove_processed_files(csv_path):
  processed_files = glob(f"{csv_path}/*/*_processed.tsv")
  for f in processed_files:
    os.remove(f)
  logger.info("Processed files removed successfully")

def df_column_to_txt(df, output_file):
    """Saves a specific text column from a DataFrame to a .txt file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in df["words"].dropna():  
            f.write(str(text) + '\n')
    logger.info("Create text files successfully")

def get_audio_duration(filepath):
  return librosa.get_duration(path=filepath)

def tsv_to_json(tsv_file, json_file):
  json_file_name =tsv_file.split("_")[0]
  data = {}
  with open(tsv_file, encoding='utf-8') as f:
      reader = csv.DictReader(f, delimiter='\t')
      for row in reader:
          uid = row['id']
          data[uid] = {
              "wav": row['wav'],#.replace('{data_root}', data_root),
              "length": float(row['length']),
              "words": row['words'],
              "domain": row['domain']
          }

  with open(json_file, 'w', encoding='utf-8') as f:
      json.dump(data, f, indent=2, ensure_ascii=False)

  logger.info(f"{json_file} successfully created!")
