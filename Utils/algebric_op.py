import numpy as np
import pandas as pd
from pandas import DataFrame


def similarity_matrix(data: DataFrame, similarity):
    """
    This function calculates the similarity matrix of a given dataset.
    :param data: The input dataset.
    :param similarity: The similarity measure used.
    :return: The similarity matrix as a Pandas DataFrame
    """
    result = []
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            result.append(similarity(data.loc[i, :].squeeze(), data.loc[j, :].squeeze()))
    return pd.DataFrame(result, columns=data.columns)


def inverse_matrix(data: DataFrame):
    """
    This function calculates the inverse of the matrix given in input.
    :param data: The matrix to inverse.
    :return: The inverted matrix.
    """
    return pd.DataFrame(np.linalg.pinv(data.values), data.columns, data.index)
