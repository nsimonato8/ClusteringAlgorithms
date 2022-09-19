import modin.pandas as pd
from modin.pandas import DataFrame
from sklearn.metrics import silhouette_score


def silhouette_tuning(A: callable, D: DataFrame, hyperpar: [], settings: dict, verbose: int = 0):
    """
    This function returns the ideal value for an hyperparameter lambda, by using the Elbow method.

    :param verbose: An integer that defines how much input must be print.
    :param settings: The list of candidates for the settings of the hyperparameters.
    :param A: The clustering Algorithm.
    :param D: The dataset, in the form of a DataFrame.
    :param hyperpar: The list of candidates for the hyperparameters.
    :return: The ideal combination of values of the hyperparameter lambda.
    """
    res = []
    scores = []
    for t in range(len(hyperpar)):
        Z = A(data=D, hyperpar=hyperpar[t], settings=settings)
        res.append(Z)
        X = (t, silhouette_score(X=Z[list(set(Z.columns) - {'cluster'})], labels=Z['cluster']))
        scores.append(X)
    if verbose == 1:
        stat = pd.Series(map(lambda x: x[1], scores),
                         index=[f"Test for {hyperpar[i]['n_clusters']}" for i in range(len(hyperpar))])
        fig = stat.plot(kind="bar", ylabel="Silhouette Score", figsize=(15, 10)).get_figure()
        fig.savefig(f'../Data/Results/TuningTests/silhouette_tuning_test_stats_KMeans.png')

    max_t = max(scores, key=lambda i: hyperpar[i[0]]['n_clusters'])
    return res[max_t[0]]
