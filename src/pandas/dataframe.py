from pandas import Series, DataFrame
from numpy import ufunc, array

def delta(data: Series, operator: ufunc) -> Series:
    return Series(
        data=operator(data.iloc[1:].to_numpy(),data.iloc[0:-1].to_numpy()),
        index=range(1, len(data)))

def to_dataframe(data: array, columns: list) -> DataFrame:
    '''
    Converts numpy array to dataframe using index and coumns from parameter.
    :param array:
    :param df:
    :return:
    '''
    return DataFrame(data, columns=columns)
