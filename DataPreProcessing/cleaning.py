from datetime import datetime

from modin.pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def date_to_features(data: DataFrame, colname: str) -> None:
    """
    This function transforms the date feature of type "str" into .
    :param colname: the date-time feature of type "str"
    :param data: The input DataFrame
    :return: None
    """
    date_names = ["month", "day", "hour", "minute", "second", "msecond"]

    def convert_date_str(date_time: str) -> (int, int, int, int, int, int, int):
        mseconds = 0
        try:
            date = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            date = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
            mseconds = date.microsecond
        return date.month, date.day, date.hour, date.minute, date.second, mseconds

    for n in range(len(date_names)):
        data.loc[:, f"{colname}_{date_names[n]}"] = data[colname].apply(lambda s: convert_date_str(s)[n])

    data.drop(columns=[colname], inplace=True)
    pass


def ip_address_to_features(data: DataFrame, colname: str) -> None:
    """
    This function explodes the IP address feature @dataframe into 4 different features.

    This transformation should keep the information contained in the IP address safe, and yield good
    results when machine learning algorithms, as shown by Enchun Shao [2019].
    :param data: The input DataFrame
    :param colname: The feature that contains the IP address.
    :return: None
    """
    for i in range(4):
        data.loc[:, f"{colname}{i + 1}"] = data[colname].apply(lambda s: int(s.split(".")[i]))

    data.drop(columns=[colname], inplace=True)


def label_encoder(data: DataFrame, colname: str) -> None:
    """
    This function converts the label of the @colname feature into numerical label.

    This function uses the ScikitLearn class LabelEncoder.
    :param data: The input DataFrame
    :param colname: The feature that needs to be transformed.
    :return: None
    """
    encoder = LabelEncoder()
    data.loc[:, colname] = encoder.fit_transform(data[colname])


def flag_to_features(data: DataFrame) -> None:
    """
    This feature converts the feature @flg into 8 different binary features, one for each flag.
    :param data:
    :return:
    """
    flag_names = ["Unknown1", "Unknown2", "Unknown3", "ACK", "PSH", "RST", "SYN", "FIN"]
    for i in range(len(flag_names)):
        if i > 2:
            data.loc[:, f"flg_{flag_names[i]}"] = data["flg"].apply(lambda s: 0 if s[i] == "." else 1)
    data.drop(columns=["flg"], inplace=True)
    pass
