import os
import pandas as pd
import numpy as np
from io import open
from xmltodict import parse as xml_parse

COLUMN_NAME_SPEED = 'Ext.Speed'

COLUMN_NAME_CADENCE = 'Cadence'

COLUMN_NAME_WATTS = 'Ext.Watts'


class Tcx:
    def __init__(self, xmldict: dict):
        self._dict = xmldict

    def to_dataframe(self) -> pd.DataFrame:
        def prepare_tcx(df: pd.DataFrame) -> pd.DataFrame:
            def first_dict_value(d: dict):
                return list(d.values()).pop()

            def find_value_by_key_containing(d: dict, key_token: str):
                first_value = [d[k] for k in d.keys() if key_token in k].pop()
                return first_value

            df['DistanceMeters'] = df['DistanceMeters'].apply(lambda x: float(x))

            df[COLUMN_NAME_WATTS] = [find_value_by_key_containing(first_dict_value(extension_dict), 'Watts') for extension_dict in df['Extensions']]
            df[COLUMN_NAME_WATTS] = df[COLUMN_NAME_WATTS].apply(lambda x: float(x))
            df[COLUMN_NAME_SPEED] = [find_value_by_key_containing(first_dict_value(extension_dict), 'Speed') for extension_dict in df['Extensions']]
            df[COLUMN_NAME_SPEED] = df[COLUMN_NAME_SPEED].apply(lambda x: float(x))

            # Distance delta
            df['DistanceMeters-delta'] = pd.Series(
                data=np.subtract(df['DistanceMeters'].iloc[1:].to_numpy(),df['DistanceMeters'].iloc[0:-1].to_numpy()),
                index=range(1, len(df)))

            # Time delta
            ## Time -> pd.Timestamp
            df['Time'] = df['Time'].apply(lambda t: pd.to_datetime(t))
            ## Time[i+1] - Time[i] type = pd.Timedelta
            df['Time-delta'] = pd.Series(
                data=np.subtract(df['Time'].iloc[1:].to_numpy(),df['Time'].iloc[0:-1].to_numpy()),
                index=range(1, len(df)))

            ## Speed [km/h] = distance [meter] / time-delta [second] * 3.6
            df['Speed'] = df['DistanceMeters-delta'] / df['Time-delta'].apply(lambda td: td.total_seconds()) * 3.6

            ## speed / cadence
            df[COLUMN_NAME_CADENCE] = df[COLUMN_NAME_CADENCE].apply(lambda x: float(x))
            df['speed-per-cadence'] = df['Speed'] / df[COLUMN_NAME_CADENCE]
            return df


        trackpoints: dict = self._dict['TrainingCenterDatabase']['Activities']['Activity']['Lap']['Track']
        list_of_trackpoint_dicts = list(trackpoints.values())[0]

        df: pd.DataFrame = pd.DataFrame.from_records(list_of_trackpoint_dicts)
        return prepare_tcx(df)


def read_tcx(file_path: str) -> Tcx:
    def read_xml(file_path: str) -> dict:
        project_root_dir = os.path.abspath('.')
        abs_file_path = os.path.join(project_root_dir, file_path)
        with open(abs_file_path, mode='r', encoding='utf-8') as f:
            content = f.read()
            return xml_parse(content)

    # read xml to dict
    return Tcx(read_xml(file_path))
