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

FILENAME = "10k_"

print("EPSILON TEST - [DBSCAN]:\nImporting data...")
test_data = pd.read_csv(f"Data/{FILENAME}sessions_cleaned.csv", sep=",", skipinitialspace=True,
                        skipfooter=3)  # Importing the sample data
n_bins = 20
x_lim = (2.3 * (10 ** 6), 2.4 * (10 ** 6))

print("Calculating distances [8_DIM_DATA]")
distances = pd.Series(np.matrix.flatten(similarity_matrix(reduce_dimensionality(data=test_data, n_final_features=8),
                                                          euclidean).to_numpy())).drop_duplicates()
print("plot(...)")
distances.plot(kind="hist", xlabel="Distance values", ylabel="Frequence",
               figsize=(35, 30), bins=n_bins, xlim=x_lim).get_figure().savefig(
    f'{FILENAME}Elbow_euclidean_similarities_8dim.png')
