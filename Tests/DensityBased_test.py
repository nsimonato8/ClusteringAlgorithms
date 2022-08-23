import datetime

import pandas as pd

from DensityBased.DB import dbscan
from Utils.distances import cosine_distance
from Utils.evaluation import silhouette_score

print("Importing data...")
timestamp = datetime.datetime.now()
test_data = pd.read_csv("../TestData/session_sample.csv", header=0)  # Importing the sample data
test_data.drop(columns="user_id", inplace=True)
print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

settings = {'iterations_max': 10}
print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

print("Clustering...")
timestamp = datetime.datetime.now()
result = dbscan(data=test_data, similarity=cosine_distance, verbose=1, settings=settings, epsilon=0.381, minpts=2)
print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

timestamp = datetime.datetime.now()
try:
    print(
        f"Score:{silhouette_score(data=result, distance=cosine_distance)}\nTime elapsed: {datetime.datetime.now() - timestamp}")
except Exception:
    print(f"Result is empty!\tTime elapsed: {datetime.datetime.now() - timestamp}")
