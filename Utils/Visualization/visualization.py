import pandas as pd
import seaborn as sns
from modin.pandas import DataFrame


def visualize_cluster(data: DataFrame, i: int = 0, h: int = 2, cluster_or_outliers: str = 'cluster',
                      additional: str = "", path: str = ""):
    sns.set_theme(style="white", palette=None)
    p = data.copy()

    sns_plot = sns.pairplot(pd.DataFrame(p), hue=cluster_or_outliers, height=h)

    if additional != "":
        additional = "[" + additional + "]"
    sns_plot.savefig(f"{path}{additional}clusters_{i}.png")
    pass


def plot_dendrogram(data: DataFrame, i: int = 0, additional: str = "", path: str = "", **kwargs):
    """
    Plots the Dendrogram of an HAC clustering result.

    :param i: A number to add at the end of the filename.
    :param data: The clustered DataFrame (with HAC).
    :param additional: The name of the file to plot.
    :param path: The path where the plot will be saved.
    :param kwargs: Additional arguments for the plot.
    :return: None
    """
    sns.set_theme(color_codes=True)
    plt = sns.clustermap(data, method='eucidean', figsize=(30, 30), row_cluster=True)
    # plt.title(f"[Experiment {i}]HAC clustering")
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig(f"{path}/{additional}_hierachical_clusters{i}.png")
