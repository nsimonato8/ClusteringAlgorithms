from ipwhois import IPWhois
from modin.pandas import DataFrame

from DataPreProcessing.cleaning import label_encoder


def format_record_for_logging(rcd: dict) -> str:
    sw = IPWhois(rcd['src_ip'])
    srcwhois = sw.lookup_rdap()
    # dw = IPWhois(rcd['dst_ip'])
    # dstwhois = dw.lookup_rdap()
    data = [
        "type:topbytes",
        "start:" + rcd['time_start'],
        "end:" + rcd['time_end'],
        "src:" + rcd['src_ip'] + ":" + rcd['src_port'],
        "srcCC:" + srcwhois['asn_country_code'],
        "srcAS:" + srcwhois['asn_description'],
        "dst:" + rcd['dst_ip'] + ":" + rcd['dst_port'],
        "proto:" + rcd['protocol'],
        "flags:" + rcd['flags'],
        "router:" + rcd['router'],
        "totbytes:" + str(round((int(rcd['ibyte']) + int(rcd['obyte']))))
    ]
    return " ".join(data)


def add_ip_lookup(data: DataFrame, colname: str) -> None:
    """
    This function integrates the input DataFrame with some data from the Whois lookup service.

    :param data: the input DataFrame
    :param colname: the tuple of the IP address split fields to integrate
    :return:
    """

    def lookup_results(ip: str) -> (str, str):
        srcwhois = IPWhois(ip).lookup_rdap()
        return srcwhois['asn_country_code'], srcwhois['asn_description']

    data.loc[:, f"{colname}_asn_country_code"] = data[colname].apply(
        lambda ip: "IT" if ip[0:6] == "192.168" else lookup_results(ip)[0])
    label_encoder(data, f"{colname}_asn_country_code")
    data.loc[:, f"{colname}_asn_description"] = data[colname].apply(
        lambda ip: "IT" if ip[0:6] == "UNIVE, IT" else lookup_results(ip)[1])
    label_encoder(data, f"{colname}_asn_description")
    pass


def add_is_priv_port(data: DataFrame, colname: str) -> None:
    """
    This functions adds a feature for the TCP ports. If the port number is privileged, it is set to 1.

    :param colname: The feature of the TCP port.
    :param data: The input DataFrame.
    :return: None
    """
    data.loc[:, f"{colname}_is_privileged"] = data[colname].apply(lambda x: 1 if x < 1024 else 0)
    pass
