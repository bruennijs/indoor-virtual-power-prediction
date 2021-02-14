from pandas import DataFrame, Series
import src.tcx as tcx

COLUMN_NAME_GEAR_RATIO = "gear-ratio"


class TrainDataSet:
    def __init__(self, t: tcx.Tcx):
        self.df = t.to_dataframe().copy()
        self.df.dropna(axis=0, subset=[tcx.COLUMN_NAME_CADENCE, tcx.COLUMN_NAME_SPEED], inplace=True)   # drop rows with nan in cad & speed
        self.df[COLUMN_NAME_GEAR_RATIO] = self.df[tcx.COLUMN_NAME_SPEED] / self.df[tcx.COLUMN_NAME_CADENCE]

    def get_dataframe(self) -> DataFrame:
        return self.df.copy()

    def cadence_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self._power)

    def cadence_to_speed(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self.df[tcx.COLUMN_NAME_SPEED])

    def speed_to_cadence(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_SPEED: self.df[tcx.COLUMN_NAME_SPEED]}), self.df[tcx.COLUMN_NAME_CADENCE])

    def cadence_to_externalspeed(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE]}), self.df[tcx.COLUMN_NAME_EXT_SPEED])

    def speed_to_power(self) -> tuple:
        power = self._power
        return (DataFrame(data={tcx.COLUMN_NAME_SPEED: self.df[tcx.COLUMN_NAME_SPEED]}), power)

    def cadence_speed_to_power(self) -> tuple:
        return (DataFrame(data={tcx.COLUMN_NAME_CADENCE: self.df[tcx.COLUMN_NAME_CADENCE],
                                tcx.COLUMN_NAME_SPEED: self.df[tcx.COLUMN_NAME_SPEED]}), self._power)
    @property
    def _power(self) -> Series:
        return self.df[tcx.COLUMN_NAME_WATTS];
