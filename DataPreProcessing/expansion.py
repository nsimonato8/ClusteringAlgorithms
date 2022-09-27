import pandas as pd

from DataPreProcessing.analysis import expand_dataset

if __name__ == "__main__":
    file_path = "../Data/sessions.csv"
    dataset = pd.read_csv(filepath_or_buffer=file_path, sep=",", skipinitialspace=True, skipfooter=3, engine='python')
    print("[BEFORE] Structure of the dataset:")
    dataset.info()

    expand_dataset(dataset)
