import os

import numpy as np
import ray
from scipy.spatial.distance import euclidean

from DataPreProcessing.importance import reduce_dimensionality
from Utils.algebric_op import similarity_matrix

# import pandas as pd

os.environ["MODIN_CPUS"] = "20"
os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
ray.shutdown()
ray.init(num_cpus=20)

import modin.pandas as pd

print("EPSILON TEST - [DBSCAN]:\nImporting data...")
test_data = pd.read_csv("Data/sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # Importing the sample data

print("Calculating distances [TEST_DATA]")
distances = pd.Series(np.matrix.flatten(similarity_matrix(test_data, euclidean).to_numpy())).drop_duplicates()
print("plot(...)")
distances.plot(kind="hist", xlabel="Distance values", ylabel="Frequence",
               figsize=(35, 30), bins=40).get_figure().savefig(f'Elbow_euclidean_similarities.png')

print("Calculating distances [8_DIM_DATA]")
distances = pd.Series(np.matrix.flatten(similarity_matrix(reduce_dimensionality(data=test_data, n_final_features=8),
                                                          euclidean).to_numpy())).drop_duplicates()
print("plot(...)")
distances.plot(kind="hist", xlabel="Distance values", ylabel="Frequence",
               figsize=(35, 30), bins=40).get_figure().savefig(f'Elbow_euclidean_similarities_8dim.png')
