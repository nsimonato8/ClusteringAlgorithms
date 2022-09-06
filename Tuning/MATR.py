"""
This file contains the implementation of the MATR algorithm, as described by Xinjie Fan et al. [2020].
The MATR algorithms returns the best hyperparameter lambda, given a certain clustering algorithm.
"""

# Input:
#     -clustering algorithm [A]       --> The clustering algorithm that will be used.
#     -similarity matrix [S]          --> A n x n matrix, where each cell S_ij = similarity(Data[i],Data[j]).
#     -candidates [{r1, · · · , rT} ] --> A list of possibile candidates for the number of present clusters.
#     -number of repetitions [J]      --> The number of iterations that the algorithm will execute.
#     -trace gap [delta]              --> The margin of error of the number of clusters identified.
import numpy as np
import pandas as pd
from pandas import DataFrame

from Utils.algebric_op import inverse_matrix, trace


def construct_clustering_matrix(data: DataFrame):
    Z = pd.DataFrame(np.zeros(shape=(data.shape[0], (data['cluster'].max() + 2))))
    for i in range(data.shape[0]):
        Z.iloc[i, data['cluster'].iloc[i] + 1] = 1
    return Z


def MATR(A, D: DataFrame, S: DataFrame, hyperpar: [], settings: dict, verbose: int = 0, name: str = "MATR"):
    """
    This function returns the ideal value for an hyperparameter lambda, by using the MATR algorithm as described by Xinjie Fan et al. [2020].
    :param name: The name of the test to put on the output file (only if verbose is 1).
    :param verbose: An integer that defines how much input must be print.
    :param settings: The list of candidates for the settings of the hyperparameters.
    :param A: The clustering Algorithm.
    :param D: The dataset, in the form of a DataFrame.
    :param S: The similarity matrix, in the form of a DataFrame.
    :param hyperpar: The list of candidates for the hyperparameters.
    :return: The ideal combination of values of the hyperparameter lambda.
    """
    res = []
    inner_prods = []
    T = len(hyperpar)
    for t in range(T):
        Z = A(data=D, hyperpar=hyperpar[t], settings=settings)
        res.append(Z)
        Z = construct_clustering_matrix(Z)
        X = S.T.dot(Z.dot(inverse_matrix(Z.T.dot(Z)).dot(Z.T)))
        inner_prods.append((t, trace(X)))
    if verbose == 1:
        stat = pd.Series(map(lambda x: x[1], inner_prods), index=[f"Test n°{i}" for i in range(len(hyperpar))])
        fig = stat.plot(kind="bar", ylabel="Trace Criterion", figsize=(15, 10)).get_figure()
        fig.savefig(f'{name}_tuning_test_stats.png')

    max_t = max(inner_prods, key=lambda i: i[1])
    return res[max_t[0]]
