import os
import warnings
from datetime import datetime

from Clustering.Hierarchical.HAC import HAC

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd

from DataPreProcessing.importance import reduce_dimensionality
from Tuning.MATR import MATR

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

master_timestamp = datetime.now()

FILENAME = ""

# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # engine='python'

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")
EXP_NUM = 1

settings_HAC = {'compute_full_tree': 'auto',
                'linkage': 'ward',
                'distance': 'euclidean',
                'epsilon': None}

data = reduce_dimensionality(test_data, 8)

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")
param = [{'n_clusters': i} for i in range(4, 20)]

print(f"[{datetime.now()}][HAC]STARTING MATR...")
timestamp1 = datetime.now()
MATR(A=HAC, D=data, hyperpar=param, settings=settings_HAC, verbose=1,
     path="Data/Results/Analysis",
     name=f"[{EXP_NUM}]]PCA_8_dim-HAC]_{FILENAME}_")
timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}][KMEANS]STARTING MATR...")
timestamp1 = datetime.now()
MATR(A=HAC, D=data, hyperpar=param, settings=settings_HAC, verbose=1,
     path="Data/Results/Analysis",
     name=f"[{EXP_NUM}][PCA_8_dim-KMEANS]_{FILENAME}_")
timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")
