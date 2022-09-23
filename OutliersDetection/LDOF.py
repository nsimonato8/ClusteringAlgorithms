"""
The algorithm computes LDOF factor on a data instance which indicate how much it deviates from its neighborhood.
Data instances obtaining high scores are more likely considered as outliers.
LDOF factor is calculated by dividing the KNN distance of an object xp by the KNN inner distance of an object xp.
This file presents an implementation of the LDOF algorithm, as described by Abir Smiti[2020].
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import modin.pandas as pd
from modin.pandas import DataFrame, Series
from sklearn.neighbors import NearestNeighbors

from Utils.algebric_op import similarity_matrix, trace


# from pandas import DataFrame, Series


def p_neighbourhood(p: Series, data: DataFrame, k: int, distance: callable) -> DataFrame:
    """
    This function retrieves the k nearest neighbours of the p instance, contained in the data DataFrame.

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours
    """
    f = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=distance, n_jobs=-1).fit(data).kneighbors(X=p.to_frame().T, n_neighbors=k, return_distance=False)  # n_jobs=-1 uses all the available processors
    return pd.DataFrame(f)


def kNN_distance(p: Series, data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the kNN distance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours distance.
    """
    kNN = p_neighbourhood(p, data, k, distance)
    return kNN.apply(lambda x: distance(p, x)).sum() / k


def kNN_inner_distance(data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the kNN inner distance, as described by Abir Smiti[2020].

    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    sim = similarity_matrix(data, distance)
    return (sim.values.sum() - trace(sim)) / (2 * k * (k - 1))


def LDOF_score(p: Series, data: DataFrame, k: int, distance: callable) -> float:
    """
    This function returns the LDOF score of a given instance, as described by Abir Smiti[2020].

    :param p: The referenced instance.
    :param data: The whole dataset
    :param k: The number of neighbours to retrieve
    :param distance: The distance function to use
    :return: The k-Nearest-Neighbours inner distance.
    """
    return kNN_distance(p, data, k, distance) / kNN_inner_distance(data, k, distance)


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

    if verbose < 1:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

    data = data.assign(LDOF=data.apply(lambda x: LDOF_score(x, data, k, distance), axis=1))
    data = data.sort_values(axis=0, by="LDOF", ascending=False)
    data.drop(["LDOF"], axis=1, inplace=True)
    data = data.assign(outlier=pd.Series([True for _ in range(n)] + [False for _ in range(n+1, data.shape[0] + 1)]))
    # data.loc[(n + 1):, "outlier"] = False
    return data
