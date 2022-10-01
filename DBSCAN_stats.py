import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from Utils.algebric_op import similarity_matrix

print("EPSILON TEST - [DBSCAN]:\nImporting data...")
test_data = pd.read_csv("Data/sessions_cleaned.csv", sep=",", skipinitialspace=True, skipfooter=3,
                        engine='python')  # Importing the sample data

print("similarity_matrix()")
distances = similarity_matrix(test_data, euclidean)
print("to_numpy()")
distances = distances.to_numpy()
print("np.matrix.flatten()")
distances = np.matrix.flatten(distances)
print("pd.Series.drop_duplicates()")
distances = pd.Series(distances).drop_duplicates()
print("plot(...)")
distances.plot(kind="bar", xlabel="Distance values", ylabel="Frequence",
               figsize=(35, 30)).get_figure().savefig(f'Elbow_euclidean_similarities.png')
