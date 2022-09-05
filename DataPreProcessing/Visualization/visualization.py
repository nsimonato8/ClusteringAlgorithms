import seaborn as sns
from pandas import DataFrame


def visualize_cluster(data: DataFrame):
    sns.set_theme(style="white", palette=None)
    sns_plot = sns.pairplot(data, hue='species', size=2.5)
    fig = sns_plot.get_figure()
    fig.savefig("clusters.png")
    pass
