import seaborn as sns
from pandas import DataFrame


def visualize_cluster(data: DataFrame, labels: DataFrame, i: int = 0, h: int = 2, additional: str = "", path: str = ""):
    sns.set_theme(style="white", palette=None)
    p = data.copy()

    p.loc[:, 'cluster'] = labels
    sns_plot = sns.pairplot(p, hue='cluster', height=h)

    if additional != "":
        additional = "[" + additional + "]"
    sns_plot.savefig(f"{path}{additional}clusters_{i}.png")
    pass
