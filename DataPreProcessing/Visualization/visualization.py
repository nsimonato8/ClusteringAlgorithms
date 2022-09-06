import seaborn as sns
from pandas import DataFrame


def visualize_cluster(data: DataFrame, labels: DataFrame, i: int = 0, h: int = 2):
    sns.set_theme(style="white", palette=None)
    p = data.copy()

    p.loc[:, 'cluster'] = labels
    sns_plot = sns.pairplot(p, hue='cluster', height=h)
    sns_plot.savefig(f"clusters{i}.png")
    pass
