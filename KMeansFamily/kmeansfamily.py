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
from datetime import datetime
from random import randint

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

import Utils.distances
import Utils.infogain
from KMeansFamily.settings import DEF_SETTINGS


def calculate_mean(cluster: DataFrame):
    """
    This function returns the instance of data that is the nearest to the mean instance of the
    :param cluster:
        This DataFrame should be a cluster.
    :return: Series
    """
    if not cluster.empty:
        mean = pd.Series(index=cluster.columns, dtype=float)
        for col in list(set(cluster.columns) - set("cluster")):
            mean[col] = cluster[col].mean()

        return mean
    else:
        return None


def kmeans(data: DataFrame, similarity=Utils.distances.euclidean_distance, infogain=Utils.infogain.has_changed,
           k: int = 5, init: str = 'random', settings: dict = DEF_SETTINGS, verbose: int = 0):
    """ 
    This functions assings a cluster number to each instance of data.
    :param data: This Dataframe contains the data to be clustered.
    :param similarity: This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1;
    :param infogain: This function returns the Information Gain Index after the last iteration. In this case, it returns true if there are changes after the iteration.
    :param k: The expected number of clusters. Default value is 5;
    :param init: The initialization method. Accepted values: ['forgy', 'random']. Default value is 'random';
    :param settings: This dictionary contains the settings for the optimization of the algorithm's performance.
    :param verbose: An integer that describes how much visual output the function prints.
    :return The initial DataFrame, but with a new column, with a progressive id for each cluster.
    """
    means = []
    iterations = 0

    timestamp = datetime.now()

    if init == "forgy":
        for _ in range(k):
            means.append(data.sample(n=1))
    else:
        data['cluster'] = pd.DataFrame([randint(1, k) for _ in range(data.shape[0])], columns=['cluster'])
        for _ in range(k):
            means.append(pd.DataFrame(columns=data.columns))

    after = data
    before = after.copy()

    while ((iterations == 0) or infogain(before, after)) and iterations < settings[
        'iterations_max']:  # The first iteration is always executed
        before = after.copy()

        if verbose == 1:
            fig = after['cluster'].plot(kind='hist').get_figure()
            plt.tight_layout()
            fig.savefig(f'test_kmeans_{iterations}.png')

        for i in range(k):  # Calculating the means for each cluster
            means[i] = calculate_mean(cluster=after.loc[after['cluster'] == i + 1])

        for i in range(after.shape[0]):
            max_similarity = 0
            for j in range(k):
                if similarity(means[j], after.iloc(axis=0)[i]) > max_similarity:
                    after.iloc[i, after.columns.get_loc("cluster")] = j + 1
                    max_similarity = similarity(means[j], after.iloc(axis=0)[i])

        if verbose == 1:
            print(
                f"\tinfogain index: {infogain(before, after)}\titerations: {iterations}\ttime elapsed: {datetime.now() - timestamp}")

        iterations += 1

    return after


# K-Medoids Custering prototype
# This prototype uses the Voronoi iteration method, known as the "Alternating" heuristic in literature.

# 1.Select initial medoids randomly
# 2.Iterate while the cost decreases:
#   a.In each cluster, make the point that minimizes the sum of distances within the cluster the medoid
#   b.Reassign each point to the cluster defined by the closest medoid determined in the previous step
def calculate_medoid(cluster: DataFrame, verbose: int, similarity):
    """
    This function returns the instance of data that is the nearest to the mean instance of the
    :param verbose: An integer that describes how much visual output the function prints.
    :param cluster: This DataFrame should be a cluster.
    :param similarity: This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1.
    :return: Series
    """
    result = None
    if not cluster.empty:
        mean = pd.Series(index=cluster.columns, dtype=float)
        for col in list(set(cluster.columns) - set("cluster")):
            mean[col] = cluster[col].mean()

        best_fit = 0
        for i in range(cluster.shape[0]):
            if similarity(mean, cluster.iloc(axis=0)[i]) > best_fit:
                result = cluster.iloc(axis=0)[i]
    else:
        return None

    return result


def kmedoids(data: DataFrame, similarity=Utils.distances.euclidean_distance, infogain=Utils.infogain.has_changed,
             k: int = 5, settings: dict = DEF_SETTINGS, verbose: int = 0):
    """
    This functions assings a cluster number to each instance of data.
    :param verbose: An integer that describes how much visual output the function prints.
    :param data: This Dataframe contains the data to be clustered.
    :param similarity: This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1;
    :param infogain: This function returns the Information Gain Index after the last iteration;
    :param k: The expected number of clusters. Default value is 5;
    :param settings: This dictionary contains the settings for the optimization of the algorithm's performance.
    :return DataFrame
    """
    medoids = []
    iterations = 0

    timestamp = datetime.now()

    data['cluster'] = pd.DataFrame([randint(1, k) for _ in range(data.shape[0])],
                                   columns=['cluster'])  # Random cluster assignment by default

    for _ in range(k):
        medoids.append(data.sample(n=1))  # Randomly assigning medoids

    after = data
    before = after.copy()

    while ((iterations == 0) or infogain(before, after)) and iterations < settings[
        'iterations_max']:  # The first iteration is always executed
        before = after.copy()

        if verbose == 1:
            fig = after['cluster'].plot(kind='hist').get_figure()
            plt.tight_layout()
            fig.savefig(f'test_kmedoids_{iterations}.png')

        for i in range(k):  # Calculating the medoids for each cluster
            medoids[i] = calculate_medoid(cluster=after.loc[after['cluster'] == i + 1], similarity=similarity,
                                          verbose=0)

        for i in range(after.shape[0]):
            max_similarity = 0
            for j in range(k):
                if similarity(medoids[j], after.iloc(axis=0)[i]) > max_similarity:
                    after.iloc[i, after.columns.get_loc("cluster")] = j + 1
                    max_similarity = similarity(medoids[j], after.iloc(axis=0)[i])

        if verbose == 1:
            print(
                f"\tinfogain index: {infogain(before, after)}\titerations: {iterations}\ttime elapsed: {datetime.now() - timestamp}")

        iterations += 1

    return after
