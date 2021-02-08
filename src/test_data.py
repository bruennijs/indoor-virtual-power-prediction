from pandas import DataFrame, Series
import src.tcx as tcx

class TrainDataSet:
    def __init__(self, tcx: tcx.Tcx):
        self.df = tcx.to_dataframe()

    def cadence_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self._get_power())

    def _get_power(self):
        return self.df[tcx.COLUMN_NAME_WATTS];
