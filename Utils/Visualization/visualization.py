import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from modin.pandas import DataFrame
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def visualize_cluster(data: DataFrame, i: int = 0, h: int = 2, cluster_or_outliers: str = 'cluster',
                      additional: str = "", path: str = ""):
    sns.set_theme(style="white", palette=None)
    p = data.copy()

    sns_plot = sns.pairplot(pd.DataFrame(p), hue=cluster_or_outliers, height=h)

    if additional != "":
        additional = "[" + additional + "]"
    sns_plot.savefig(f"{path}{additional}clusters_{i}.png")
    pass


def plot_dendrogram(model: AgglomerativeClustering, i: int = 0, additional: str = "", path: str = "", **kwargs):
    """
    Plots the Dendrogram of an HAC clustering result.

    :param i: A number to add at the end of the filename.
    :param model: The fitted model to plot.
    :param additional: The name of the file to plot
    :param path: The path where the plot will be saved.
    :param kwargs: Additional arguments for the plot.
    :return: None
    """
    # Create linkage matrix and then plot the dendrogram

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.title(f"[Experiment {i}]HAC clustering")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(f"{path}/{additional}_hierachical_clusters{i}.png")
