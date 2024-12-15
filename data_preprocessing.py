import pandas as pd
from glob import glob
import os

# paths to the data files
audio_path = os.path.join(os.getcwd(),"lozgen/audio/")
csv_path = os.path.join(os.getcwd(),"lozgen/splits/")  

def prepare_data(audio_path, csv_path):
    split_list = ["male","female","combined"]
    for split in split_list:
        csv_file_list = glob(f"{csv_path}{split}/*.tsv")
        for csv_file in csv_file_list:
            split_file = os.path.basename(csv_file).split(".")[0]
            df = pd.read_csv(csv_file, sep="\t")
            df["path"] = audio_path + df['audio']
            df = df.dropna(subset=["path"])
            df = df.drop(columns=['audio'])
            df = df.rename(columns={'path':'audio'})
            df = df[["audio","sentence"]]
            df.to_csv(f"{csv_path}/{split}/{split_file}_processed.tsv", sep="\t", index=False)
            print(f"{split_file}_processed : ", len(df))

if __name__== "__main__":
	prepare_data(audio_path, csv_path) 