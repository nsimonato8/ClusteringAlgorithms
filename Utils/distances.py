from math import sqrt

from pandas import Series


def euclidean_distance(x: Series, y: Series):
    """
    Simple function that returns the euclidean distance beetween two vectors.
    :param x: Vector 1.
    :param y: Vector 2.
    :return: The euclidean distance beetween Vector 1 and Vector 2. For more info, see:  https://en.wikipedia.org/wiki/Euclidean_distance
    """
    if x.empty or y.empty:
        return float(not x.empty and not y.empty)

    return sqrt((x ** 2 + y ** 2).sum(axis=0))


def cosine_distance(x: Series, y: Series):
    """
    Simple function that returns the cosine similarity beetween two vectors.
    :param x: Vector 1.
    :param y: Vector 2.
    :return: The cosine similarity beetween Vector 1 and Vector 2. For more info, see: https://en.wikipedia.org/wiki/Cosine_similarity
    """
    to_exclude = ["a_index", "b_index", "max_index", "cluster"]
    if x.empty or y.empty:
        return float(not x.empty and not y.empty)
    x = x[list(set(x.index) - set(to_exclude))]
    y = y[list(set(y.index) - set(to_exclude))]
    return x.dot(y) / sqrt((x ** 2).sum()) / sqrt((y ** 2).sum())
