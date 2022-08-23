import datetime

import pandas as pd

import KMeansFamily.kmeansfamily
import Utils.distances
import Utils.evaluation
import Utils.infogain

print("Importing data...")
timestamp = datetime.datetime.now()
test_data = pd.read_csv("../TestData/session_sample.csv")  # Importing the sample data
print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

settings = {'iterations_max': 10}
print(f"Settings:\n\tMaximum number of iterations: {settings['iterations_max']}")

print("Clustering...")
timestamp = datetime.datetime.now()
result = KMeansFamily.kmeansfamily.kmedoids(data=test_data, similarity=Utils.distances.cosine_distance,
                                            infogain=Utils.infogain.has_changed, verbose=1)
print(f"\tTime elapsed:\t{datetime.datetime.now() - timestamp}")

timestamp = datetime.datetime.now()
print(
    f"Result:\n{result.info()}\nScore:{Utils.evaluation.silhouette_score(data=result, distance=Utils.distances.cosine_distance)}\nTime elapsed: {datetime.datetime.now() - timestamp}")
