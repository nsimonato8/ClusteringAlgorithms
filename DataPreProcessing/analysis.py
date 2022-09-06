import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from DataPreProcessing.cleaning import date_to_features, ip_address_to_features, label_encoder, flag_to_features
from DataPreProcessing.feature_eng import add_is_priv_port


def plot_data_distribution(data: DataFrame) -> None:
    """
    This function plots the
    :param data:
    :return:
    """
    try:
        # create the figure and axes
        fig, axes = plt.subplots(int(data.shape[1] / 2), 2, figsize=(25, 25))

        # unpack all the axes subplots
        axe = axes.ravel()

        # assign the plot to each subplot in axe
        for i, c in enumerate(data.columns):
            data[c].plot(kind="hist", x=c, ax=axe[i])

        plt.tight_layout()
        fig.savefig(f'../Data/Results/SessionDataDistribution/features_distribution_cleaned.png')
    except TypeError as t:
        print(f"Cannot print plots: {t}")
    except AttributeError as a:
        print(f"Cannot print plots: {a}")


def clean_dataset(data: DataFrame) -> None:
    """
    This function encapsulate all the cleaning operations conducted in the pre-processing phase.
    :param data: The input DataFrame
    :return: None
    """
    to_drop_mpls = [f"mpls{i}" for i in range(1, 11)]
    to_drop = to_drop_mpls + ["smk", "dmk", "nh", "nhb", "svln", "dvln", "ismc", "odmc", "idmc", "osmc", "cl", "sl",
                              "al", "eng", "exid", "ra", "td", "in", "out"]

    data.drop(columns=to_drop, inplace=True)
    pass


def transform_dataset(data: DataFrame) -> None:
    """
    This function encapsulate all the data transformation operations executed on the dataset.
    :param data: The input DataFrame
    :return: None
    """
    # Converting Datetime
    date_to_features(data, "ts")
    date_to_features(data, "te")
    date_to_features(data, "tr")

    # Transforming IP addresses
    ip_address_to_features(data, "sa")
    ip_address_to_features(data, "da")

    # Encoding labels
    label_encoder(data, "pr")

    # Transforming flg
    flag_to_features(data)
    pass


def expand_dataset(data: DataFrame) -> None:
    """
    This function encapsulate all the expansions operations executed on the dataset.
    :param data: The input DataFrame
    :return: None
    """
    # signaling if the port is privileged
    add_is_priv_port(data, "sp")
    add_is_priv_port(data, "dp")
    pass


if __name__ == "__main__":
    file_path = "../Data/10k_sessions.csv"
    dataset = pd.read_csv(filepath_or_buffer=file_path, sep=",", skipinitialspace=True, skipfooter=3, engine='python')
    print("[BEFORE] Structure of the dataset:")
    dataset.info()

    clean_dataset(dataset)

    print("[AFTER DROPPING] Structure of the dataset:")
    dataset.info()
    print(dataset.head(n=15))

    transform_dataset(dataset)

    print("[AFTER TRANSFORMING] Structure of the dataset:")
    dataset.info()
    print(dataset.head(n=15))

    to_print = ["sp", "dp"]  # , "stos", "dtos", "fwd", "ipkt", "opkt", "ibyt", "obyt", "pr"]

    plot_data_distribution(dataset[to_print])

    expand_dataset(dataset)

    print("[AFTER EXPANDING] Structure of the dataset:")
    dataset.info()
    print(dataset.head(n=15))

    dataset.to_csv("../Data/10k_sessions_cleaned.csv", index=False)
