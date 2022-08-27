from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from DensityBased.settings import DEF_SETTINGS


#  1. Mark all the objects as unvisited;
#  2. repeat:
#  3.    randomly select an unvisited object p;
#  4.    mark p as visited;
#  5.    if the e-neighbourhood of p has at least MinPts objects:
#  6.        create a new cluster C, and add p to C;
#  7.        let N be the set of objects in the e-neighbourhood of p;
#  8.        for each point p' in N:
#  9.            if p' is unvisited:
# 10.                mark p' as visited;
# 11.                if the e-neighbourhood of p' has at least MinPts points:
# 12.                    add those points to N;
# 13.           if p' is not yet a member of any cluster, add p' to C;
# 14.        end for;
# 15.        output C
# 16.   else:
# 17.       mark p as noise;
# 18. until unvisited is empty;


def neighbourhood(data: DataFrame, point: DataFrame, epsilon: float, similarity):
    """
    This functions returns the neighbourhood of point, given that points is contained in data.

    The neighbourhood of point is defined as the set of points at maximum distance (intended as similarity distance) epsilon.
    :param data: The input DataFrame.
    :param point: The currently visited point.
    :param epsilon: The radius parameter.
    :param similarity: This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1.
    :return: DataFrame
    """
    assert point.shape[0] != 1, "The inserted point is not valid"

    result = data.copy()
    # point['p_similarity'] = 0.0
    result['p_similarity'] = data.apply(lambda x: similarity(x, point), axis=1)
    result = result.loc[result['p_similarity'] <= epsilon]
    result.drop(columns=["p_similarity"], axis=1, inplace=True)
    return result


def dbscan(data: DataFrame, epsilon: float, minpts: int, similarity, settings: dict = DEF_SETTINGS, verbose: int = 0):
    """
    Implementation of the DB-Scan algorithm. For more information, see: https://en.wikipedia.org/wiki/DBSCAN
    :param data:  The input DataFrame.
    :param epsilon: The radius parameter.
    :param minpts: The neighbourhood density treshold.
    :param similarity: This function returns the similarity index between the two instances of the considered data. It must be a float between 0 and 1.
    :param settings: This dictionary contains the settings for the optimization of the algorithm's performance.
    :param verbose: An integer that describes how much visual output the function prints.
    :return: DataFrame
    """
    clusters_count = 0
    iterations = 0
    timestamp = None

    # Initialization of the dataframe
    data.loc[:, 'cluster'] = 0  # All points are set as "noise" by default
    data.loc[:, 'visited'] = False  # All points are marked as unvisited

    while True:
        if verbose == 1:
            timestamp = datetime.now()

        p = data.sample(n=1, replace=False)
        data.iloc[p.index, data.columns.get_loc("visited")] = True
        p = p.squeeze()
        p_neighbourhood = neighbourhood(data=data, point=p, epsilon=epsilon, similarity=similarity)
        if p_neighbourhood.shape[0] >= minpts:
            clusters_count += 1
            for i in range(p_neighbourhood.shape[0]):
                if not p_neighbourhood.iloc[i, p_neighbourhood.columns.get_loc("visited")]:
                    p_neighbourhood.iloc[i, p_neighbourhood.columns.get_loc("visited")] = True
                    q_neighbourhood = neighbourhood(data=data, point=p_neighbourhood.iloc[i], epsilon=epsilon,
                                                    similarity=similarity)
                    if q_neighbourhood.shape[0] > minpts:
                        p_neighbourhood = pd.concat([p_neighbourhood, q_neighbourhood], axis=0)
                    if p_neighbourhood.iloc[i]['cluster'] == 0:
                        data.loc[p_neighbourhood.iloc[i, p_neighbourhood.columns.get_loc("index")] == data[
                            'index'], "cluster"] = clusters_count  # TODO: Find out how to communicate the index among copies of the dataframe

        if verbose == 1:
            print(
                f"\tIteration {iterations}:\tTime elapsed:{datetime.now() - timestamp}\tClusters_count: {clusters_count}")
            fig = data['cluster'].plot(kind='hist').get_figure()
            plt.tight_layout()
            fig.savefig(f'test_DBSCAN_{iterations}.png')
        iterations += 1

        if data.loc[data['visited'] == False].empty or iterations == settings['iterations_max']:
            break

    return data
