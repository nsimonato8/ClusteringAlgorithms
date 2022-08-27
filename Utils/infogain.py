from math import log2

from pandas import Series


def has_changed(before, after):
    """
    This function returns a naive index of the state change of the "before" DataFrame.

    The result is just the sum of the absolute values of the numeric differences beetween before['cluster'] and
    after['cluster'], applied to the inverse logit function for standardization.
    :param before: Pandas DataFrame that represents the state of the data before the iteration.
    :param after: Pandas DataFrame that represents the state of the data after the iteration.
    :return: A boolean that 's True iff before and after are different.
    """

    assert before['cluster'].size == after['cluster'].size and before['cluster'].size > 0 and after[
        'cluster'].size > 0, "One of the inputs is incorrect"

    return not before['cluster'].equals(after['cluster'])


def entropy(data: Series):
    """
    Utility function that calculates the entropy of a Pandas Series.

    The entropy formula refers to this page: https://en.wikipedia.org/wiki/Entropy_(information_theory).
    :param data: The input Pandas Series
    :return: The entropy of the Pandas Series, expressed as a float beetween 0 and 1.
    """
    if data.empty:
        return 0
    else:
        p = data.value_counts(normalize=True)
        p = p.apply(lambda x: -x * log2(x))
        return p.sum()


def var_entropy(before, after):
    """
    This function returns the variation of entropy from the "before" DataFrame to the "after" DataFrame.

    The result is just the sum of the absolute values of the numeric differences beetween before['cluster'] and
    after['cluster'], applied to the inverse logit function for standardization.
    :param before: Pandas DataFrame that represents the state of the data before the iteration.
    :param after: Pandas DataFrame that represents the state of the data after the iteration.
    :return: A float index beetween 0 and 1.
    """

    assert before['cluster'].size == after['cluster'].size and before['cluster'].size > 0 and after[
        'cluster'].size > 0, "One of the inputs is incorrect"

    return entropy(after['cluster']) - entropy(before['cluster'])
