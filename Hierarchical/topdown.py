# K-Means Clustering prototype

# 0.Initialization:
#   a.The Forgy method randomly chooses k observations from the dataset and uses these as the initial means.
#   b.The Random Partition method first randomly assigns a cluster to each observation and then proceeds to the update step,
#     thus computing the initial mean to be the centroid of the cluster's randomly assigned points.
# 1.Given an initial set of k means m1(1),...,mk(1):
# 2.Assignment step:
#   a.Assign each observation to the cluster with the nearest mean: that with the least squared Euclidean distance.
# 3.Update step:
#   a.Recalculate means (centroids) for observations assigned to each cluster.
# 4.The algorithm has converged when the assignments no longer change.
from random import randint

import pandas as pd
from pandas import DataFrame


def calculate_mean(data: DataFrame, similarity):
    """
    This function returns the instance of data that is the nearest to the mean instance of the
    :param data:
        This DataFrame should be a cluster.
    :param similarity:
        This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1.
    :return: DataFrame
    """
    mean = pd.DataFrame()
    result = None
    for col in list(set(data.columns) - set("cluster")):
        mean[col] = data[col].mean()

    best_fit = 0
    for i in range(data.size):
        if similarity(mean, data.loc[i]) > best_fit:
            result = data.loc[i]

    return result


def kmeans(data: DataFrame, similarity, infogain, k: int = 5, init: str = 'random',
           settings: dict = {'iterations_max': 1000, 'min_i_treshold': 0.005}):
    """ 
    This functions assings a cluster number to each instance of data.

        :param data:
            This Dataframe contains the data to be clustered.
        :param similarity:
            This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1;
        :param infogain:
            This function returns the Information Gain Index after the last iteration;
        :param k:
            The expected number of clusters. Default value is 5;
        :param init:
            The initialization method. Accepted values: ['forgy', 'random']. Default value is 'random';
        :param settings:
            This dictionary contains the settings for the optimization of the algorithm's performance.
    """
    means = [None] * k
    iterations = 0
    if not init == "forgy" or init == "random":
        for i in range(data.size):
            data['cluster'] = randint(1, k)

    if init == "forgy":
        for i in range(k):
            means[i] = data.sample(n=1)

    while infogain(data) > settings['min_i_treshold'] and iterations < settings['iterations_max']:
        if iterations == 0 and init != "forgy":
            for i in range(k):  # Calculating the means for each cluster
                means[i] = calculate_mean(data.loc[data['cluster'] == i], similarity)
        for instance in data:
            max_similarity = 0
            for i in range(k):
                if similarity(means[i], instance) > max_similarity:
                    instance['cluster'] = i

        iterations += 1

    return data

# K-Medoids Custering prototype
# This prototype uses the Voronoi iteration method, known as the "Alternating" heuristic in literature.

# 1.Select initial medoids randomly
# 2.Iterate while the cost decreases:
#   a.In each cluster, make the point that minimizes the sum of distances within the cluster the medoid
#   b.Reassign each point to the cluster defined by the closest medoid determined in the previous step
