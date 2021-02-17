from pandas import Series
from numpy import ufunc

def delta(data: Series, operator: ufunc) -> Series:
    return Series(
        data=operator(data.iloc[1:].to_numpy(),data.iloc[0:-1].to_numpy()),
        index=range(1, len(data)))
