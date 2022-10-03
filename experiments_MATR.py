import os
import warnings
from datetime import datetime

from Clustering.Hierarchical.HAC import HAC
from Tuning.silhouette import silhouette_tuning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray
# import pandas as pd

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

# test_data = reduce_dimensionality(test_data, 8)

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")
param = [{'n_clusters': i} for i in range(2, 30)]

print(f"[{datetime.now()}][HAC]STARTING MATR...")
timestamp1 = datetime.now()
hac_matr = MATR(A=HAC, D=test_data, hyperpar=param, settings=settings_HAC, verbose=1,
                path="Data/Results/Analysis",
                name=f"[Full_dim-HAC]MATR_{FILENAME}_")
timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}][HAC]STARTING Silhouette...")
timestamp1 = datetime.now()
hac_silh = silhouette_tuning(A=HAC, D=test_data, hyperpar=param, settings=settings_HAC, verbose=1,
                             path="Data/Results/Analysis",
                             name=f"[Full_dim-HAC]SIL_{FILENAME}_")
timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")
