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

import modin.pandas as pd
from modin.pandas import DataFrame

from Utils.algebric_op import inverse_matrix, trace, construct_clustering_matrix, similarity_matrix


def MATR(A: callable, D: DataFrame, hyperpar: [], settings: dict, verbose: int = 0, name: str = "MATR", path: str = "",
         output: int = 0):
    """
    This function returns the ideal value for an hyperparameter lambda, by using the MATR algorithm as described by Xinjie Fan et al. [2020].

    :param path: The path where the verbose output is saved.
    :param output: If set to 0, returns the clustering with the best result. If set to 1, returns all the clustering results and the best one.
    :param name: The name of the test to put on the output file (only if verbose is 1).
    :param verbose: An integer that defines how much input must be print.
    :param settings: The list of candidates for the settings of the hyperparameters.
    :param A: The clustering Algorithm.
    :param D: The dataset, in the form of a DataFrame.
    :param hyperpar: The list of candidates for the hyperparameters.
    :return: The ideal combination of values of the hyperparameter lambda.
    """
    S = similarity_matrix(D, settings['distance'])
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
        stat = pd.Series(map(lambda x: x[1], inner_prods),
                         index=[f"Test for k={hyperpar[i]['n_clusters']}" for i in range(len(hyperpar))])
        fig = stat.plot(kind="bar", xlabel="Number of clusters", ylabel="Trace Criterion",
                        figsize=(15, 10)).get_figure()
        fig.savefig(f'{path}{name}_tuning_test_stats.png')

    max_t = max(inner_prods, key=lambda i: i[1])
    # if output == 1:
    #     return res, res[max_t[0]]
    # else:
    return res[max_t[0]]
