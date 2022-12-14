import os
import sys
import warnings
from datetime import datetime

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV

from DataPreProcessing.importance import reduce_dimensionality
from Utils.Visualization.visualization import visualize_cluster

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import ray

# import pandas as pd

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
# os.environ["MODIN_MEMORY"] =
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

master_timestamp = datetime.now()
EXP_NUM = 7
FILENAME = "10k_"
# Importing Data
print(f"[{datetime.now()}]IMPORTING DATA...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # engine='python'

# Settings
print(f"[{datetime.now()}]GENERATING SETTINGS...")

settings_GridSearch = {'estimator': DBSCAN(),
                       'refit': True,
                       'verbose': 3,
                       'return_train_score': True,
                       'scoring': 'adjusted_mutual_info_score',
                       'cv': 3,
                       'error_score': 1,
                       }
settings_DBSCAN = {'eps': [x for x in np.arange(2.34 * (10 ** 6), 2.36 * (10 ** 6), 1000.)],
                   'min_samples': [x for x in range(1, 21, 2)],
                   'metric': [euclidean],
                   'algorithm': ['auto'],
                   'n_jobs': [-1]}

print(f"[{datetime.now()}]GENERATING HYPERPARAMETERS CANDIDATES...")

print(f"[{datetime.now()}]STARTING GridSearch...")
timestamp1 = datetime.now()
model = GridSearchCV(estimator=settings_GridSearch['estimator'],
                     refit=settings_GridSearch['refit'],
                     verbose=settings_GridSearch['verbose'],
                     return_train_score=settings_GridSearch['return_train_score'],
                     scoring=settings_GridSearch['scoring'],
                     cv=settings_GridSearch['cv'],
                     error_score=settings_GridSearch['error_score'],
                     param_grid=settings_DBSCAN)

data_aux = reduce_dimensionality(data=test_data, n_final_features=8)

model.fit(data_aux)

best_dbscan_settings = model.best_params_

timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")
print(f"\tModel Parameters:\t{model.best_params_}")

print(f"[{datetime.now()}]ASSIGNING DBSCAN LABELS...")

selected_model = DBSCAN(eps=best_dbscan_settings['eps'],
                        min_samples=best_dbscan_settings['min_samples'],
                        n_jobs=-1,
                        algorithm=best_dbscan_settings['algorithm'])
aux1 = data_aux.assign(cluster=selected_model.fit_predict(X=data_aux))

timestamp1 = datetime.now() - timestamp1
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp1}...")

print(f"[{datetime.now()}]OUTLIER DETECTION (with 8 dimensions)...")
print(f"[{datetime.now()}]{'=' * 5} DBSCAN labels {'=' * 5}")
timestamp2 = datetime.now()

det_dbscan = aux1.assign(outlier=aux1['cluster'].apply(lambda x: 1 if x < 0 else 0))

print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp2}...")
print(f"[{datetime.now()}]{'=' * 5}---{'=' * 5}")

print(f"[{datetime.now()}]PRINTING CLUSTERING PAIRPLOTS TO FILES...")
# Printing the clusters
timestamp3 = datetime.now()

print(f"\t[{datetime.now()}]\tCLUSTER PLOT...")
visualize_cluster(data=aux1,
                  i=EXP_NUM,
                  cluster_or_outliers='cluster',
                  additional=f"PCA_{len(aux1.columns) - 1}_dim-DBSCAN_{aux1['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
                  path="Data/Results/Experiments/DBSCAN/")

print(f"\t[{datetime.now()}]\tOUTLIER PLOT...")
visualize_cluster(data=det_dbscan[list(set(det_dbscan.columns) - {'cluster'})],
                  i=EXP_NUM,
                  cluster_or_outliers='outlier',
                  additional=f"[DBSCAN]PCA_{len(det_dbscan.columns) - 1}_dim-DBSCAN_{det_dbscan['cluster'].max() + 1}_{FILENAME}{EXP_NUM}",
                  path="Data/Results/Experiments/DBSCAN/")

timestamp3 = datetime.now() - timestamp3
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp3}...")

print(f"[{datetime.now()}]PRINTING RESULTS TO FILES...")
timestamp7 = datetime.now()

path_results = "Data/Results/Experiments/DBSCAN/"

det_dbscan.to_csv(path_results + f"DBSCAN_Outliers_Filtering{FILENAME}{EXP_NUM}.csv")

timestamp7 = datetime.now() - timestamp7
print(f"[{datetime.now()}]DONE! Time elapsed:\t{timestamp7}...")

master_timestamp = datetime.now() - master_timestamp
print(f"[{datetime.now()}]EXPERIMENT {EXP_NUM} CONCLUDED! Time elapsed:\t{master_timestamp}...")

original_stdout = sys.stdout
with open(f'Data/Results/Experiments/DBSCAN/[Experiment PCA-DBSCAN-GridSearchCV]{FILENAME}_main_log_{EXP_NUM}.txt',
          'w') as f:
    sys.stdout = f
    print(f"GridSearchCV settings:\t{settings_GridSearch}")
    print(f"DBSCAN settings:\t{settings_DBSCAN}")
    print(f"Selected model parameters:\t{model.best_params_}")
    print(f"Time elapsed for GridSearchCV computation (8 DIM):\t{timestamp1}")
    print(f"Time elapsed for DBSCAN labeling:\t{timestamp2}")
    print(f"Time elapsed for Plotting:\t{timestamp3}")
    print(f"Total Time:\t{master_timestamp}")
sys.stdout = original_stdout
