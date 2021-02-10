from pandas import DataFrame, Series
import src.tcx as tcx

class TrainDataSet:
    def __init__(self, t: tcx.Tcx):
        self.df = t.to_dataframe()
        self.df.dropna(axis=0, subset=[tcx.COLUMN_NAME_CADENCE, tcx.COLUMN_NAME_SPEED], inplace=True)   # drop rows with nan in cad & speed

    def get_dataframe(self):
        return self.df

    def cadence_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self._get_power())

    def cadence_to_speed(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self.df[tcx.COLUMN_NAME_SPEED])

    def speed_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_SPEED: self.df[tcx.COLUMN_NAME_SPEED]}), self._get_power())

    def cadence_speed_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE],
                                tcx.COLUMN_NAME_SPEED: self.df[tcx.COLUMN_NAME_SPEED]}), self._get_power())

    def _get_power(self):
        return self.df[tcx.COLUMN_NAME_WATTS];
