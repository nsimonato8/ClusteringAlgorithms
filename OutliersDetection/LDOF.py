"""
The algorithm computes LDOF factor on a data instance which indicate how much it deviates from its neighborhood.
Data instances obtaining high scores are more likely considered as outliers.
LDOF factor is calculated by dividing the KNN distance of an object xp by the KNN inner distance of an object xp.
This file presents an implementation of the LDOF algorithm, as described by Abir Smiti[2020].
"""
import warnings
from datetime import datetime

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import modin.pandas as pd
from modin.pandas import DataFrame, Series
from sklearn.neighbors import NearestNeighbors

from Utils.algebric_op import similarity_matrix, trace


# from pandas import DataFrame, Series


def p_neighbourhood(p: DataFrame, data: DataFrame, k: int, distance: callable) -> Series:
    """
    This function retrieves the k nearest neighbours of the p instance, contained in the data DataFrame.

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours
    """
    # print(f"\t[p]\n{p.info()}")
    return NearestNeighbors(n_neighbors=k, algorithm='auto', metric=distance, n_jobs=-1).fit(data).kneighbors(X=p, n_neighbors=k, return_distance=True)[1]  # n_jobs=-1 uses all the available processors


def kNN_distance(p: Series, data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the kNN distance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours distance.
    """
    kNN = p_neighbourhood(pd.DataFrame(np.reshape(p.values, (1, data.shape[1]))), data, k, distance)
    # print(f"[.values]{kNN.values}\t[.sum]{kNN.values.sum()}")
    return kNN.sum() / k


def kNN_inner_distance(sim: DataFrame, k: int) -> float:
    """
    This function returns the kNN inner distance, as described by Abir Smiti[2020].

    :param sim: The similarity matrix
    :param k: The number of neighbours to retrieve
    :return: The k-Nearest-Neighbours inner distance.
    """
    return (sim.values.sum() - trace(sim)) / (2 * k * (k - 1))


def LDOF_score(p: Series, data: DataFrame, k: int, distance: callable, sim: DataFrame) -> float:
    """
    This function returns the LDOF score of a given instance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :param sim: The similarity matrix
    :return: The k-Nearest-Neighbours inner distance.
    """
    # dist = kNN_distance(p, data, k, distance)
    # inner_dist = kNN_inner_distance(sim, k)
    # print(f"[kNN_distance]{dist}\t[kNN_inner_distance]{inner_dist}")
    # return dist / inner_dist
    return kNN_distance(p, data, k, distance) / kNN_inner_distance(sim, k)


def top_n_LDOF(data: DataFrame, distance: callable, n: int, k: int, verbose: int = 0) -> DataFrame:
    """
    This function implements the Top-n LDOF algorithm, as described by Abir Smiti[2020].

    :param verbose: The amount of output to visualize (with 0 the warnings are suppressed)
    :param n: The number of outliers to retrieve
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    assert k > 0, "The number of neighbours must be >= 1"
    assert n > 0, "The number of outliers to retrieve must be >= 1"

    if verbose in [0, 1]:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

    sim = similarity_matrix(data, distance)
    data = data.assign(LDOF=data.apply(lambda x: LDOF_score(x, data, k, distance, sim), axis=1))

    print(f"[{datetime.now()}]\n{data['LDOF'].describe()}\n") if verbose == 1 else 0

    data.sort_values(by="LDOF", ascending=False, inplace=True)
    data.drop(["LDOF"], axis=1, inplace=True)

    data = data.assign(outlier=pd.Series([1 for _ in range(n)] + [0 for _ in range(n+1, data.shape[0] + 1)]))

    print(f"[{datetime.now()}]\n{data.get('outlier', default='outlier is missing')}\n") if verbose == 1 else 0

    return data
