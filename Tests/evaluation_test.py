from random import randint

import pandas as pd

import Utils.evaluation
from Utils.distances import euclidean_distance

test_data = [(randint(0, 10), randint(0, 10), randint(0, 10)) for i in range(0, 100)]
test_data = pd.DataFrame(data=test_data, columns=["a", "b", "cluster"])

print(f"Test data: \n{test_data}\n")

print(
    f"Silhouette index rating: {Utils.evaluation.silhouette_score(data=test_data, distance=euclidean_distance)}")
