"""
This file contains the implementation of the MATR-CV algorithm, as described by Xinjie Fan et al. [2020].
The MATR-CV algorithms returns the number of clusters that are contained in the data, given a certain clustering algorithm.
"""

# Input:
#     -clustering algorithm [A]       --> The clustering algorithm that will be used.
#     -similarity matrix [S]          --> A n x n matrix, where each cell S_ij = similarity(Data[i],Data[j]).
#     -candidates [{r1, · · · , rT} ] --> A list of possibile candidates for the number of present clusters.
#     -number of repetitions [J]      --> The number of iterations that the algorithm will execute.
#     -training ratio [y_train]       --> The proportion of the data that will be used for the training of the algorithm.
#     -trace gap [delta]              --> The margin of error of the number of clusters identified.
from statistics import median

import numpy as np
import pandas as pd
from pandas import DataFrame

from Utils.algebric_op import construct_clustering_matrix, inverse_matrix, trace


def nodesplitting(S: DataFrame, n: int, y_train: float):
    return S.iloc[:(n * y_train), :(n * y_train)], S.iloc[(n * y_train):, :(n * y_train)], S.iloc[(n * y_train):,
                                                                                           (n * y_train):]


def clustertest(S: DataFrame, Z: DataFrame):
    Z2 = Z.T.dot(Z)
    Z2 = pd.DataFrame(np.linalg.pinv(Z2.values), Z2.columns, Z2.index)
    M = S.dot(Z.dot(Z2))
    Z22 = pd.Dataframe()
    for i in range(S.shape[0]):
        Z22.iloc[i, M.iloc[i, :].max()] = 1  # Zˆ22(i, arg max M(i,:)) = 1
    return Z22


def matrcv(A, S: DataFrame, r: [], J: int, y_train: float, delta: float, **settings):
    """
    This function returns the number of clusters present in the training data. It is indicated for clustering algorithms
    such as k-means and k-medoids.
    :param A: The clustering algorithm.
    :param S: The nxn similarity matrix of the data.
    :param r: A Pandas Series of the possible value of the number of clusters.
    :param J: The number of iterations that the algorithm should do.
    :param y_train: The proportion of data to use as training data for the algorithm.
    :param delta: A parameter.
    :param settings: The settings to use for the clustering algorithm.
    :return: An integer, the number of clusters present in the data.
    """
    result = []

    for j in range(J):
        L = []
        for t in range(r.size):
            S11, S21, S22 = nodesplitting(S, S.shape[0], y_train)
            Z11 = A(S11, r[t], settings)
            Z11 = construct_clustering_matrix(Z11)
            Z22 = clustertest(S21, Z11)
            Z_i = inverse_matrix(Z11.T.dot(Z11))
            X22 = Z22.dot(Z_i.dot(Z22.T))
            L.append((t, trace(S22.T.dot(X22))))
        # result.append()  # r*j = min{rt : lrt,j ≥ maxt lrt,j − ∆}
    return median(result)
