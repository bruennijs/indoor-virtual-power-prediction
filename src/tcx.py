import os
import pandas as pd
import numpy as np
from io import open
from xmltodict import parse as xml_parse

from src.pandas.dataframe import delta

COLUMN_NAME_CADENCE_RATE = 'Cadence-rate'

COLUMN_NAME_DELTA_T = 'Time-delta'

COLUMN_NAME_EXT_SPEED = 'Ext.Speed'
COLUMN_NAME_ACCELERATION = 'Acceleration'
COLUMN_NAME_SPEED = 'Speed'
COLUMN_NAME_CADENCE = 'Cadence'
COLUMN_NAME_WATTS = 'Ext.Watts'


class Tcx(object):
    def __init__(self, xmldict: dict):
        self._dict = xmldict

    def to_dataframe(self) -> pd.DataFrame:
        def prepare_tcx(df: pd.DataFrame) -> pd.DataFrame:
            """
            Used if key is unknwon or differs per TCX implementat
            :param df: ion like <TCX> with or without namespace
            :return:
            """
            def first_dict_value(d: dict):
                return list(d.values()).pop()

            def find_value_by_key_containing(d: dict, key_token: str):
                first_value = [d[k] for k in d.keys() if key_token in k].pop()
                return first_value

            df['DistanceMeters'] = df['DistanceMeters'].apply(lambda x: float(x))

            df[COLUMN_NAME_WATTS] = [find_value_by_key_containing(first_dict_value(extension_dict), 'Watts') for extension_dict in df['Extensions']]
            df[COLUMN_NAME_WATTS] = df[COLUMN_NAME_WATTS].apply(lambda x: float(x))
            df[COLUMN_NAME_EXT_SPEED] = [find_value_by_key_containing(first_dict_value(extension_dict), 'Speed') for extension_dict in df['Extensions']]
            df[COLUMN_NAME_EXT_SPEED] = df[COLUMN_NAME_EXT_SPEED].apply(lambda x: float(x))

            # Distance delta
            df['DistanceMeters-delta'] = delta(df['DistanceMeters'], np.subtract)

            # Time delta
            ## Time -> pd.Timestamp
            df['Time'] = df['Time'].apply(lambda t: pd.to_datetime(t))
            ## Time[i+1] - Time[i] type = pd.Timedelta
            df[COLUMN_NAME_DELTA_T] = delta(df['Time'], np.subtract)

            ## Speed [km/h] = distance [meter] / time-delta [second] * 3.6
            df[COLUMN_NAME_SPEED] = (df['DistanceMeters-delta'] / df[COLUMN_NAME_DELTA_T].apply(lambda td: td.total_seconds())) * 3.6

            # delta Speed
            speed_delta: pd.Series = delta(df[COLUMN_NAME_SPEED], np.subtract)
            df[COLUMN_NAME_ACCELERATION] = (speed_delta / df[COLUMN_NAME_DELTA_T].apply(lambda td: td.total_seconds())) * 3.6

            ## cadence
            df[COLUMN_NAME_CADENCE] = df[COLUMN_NAME_CADENCE].apply(lambda x: float(x))

            # delta Cadence
            cadence_delta: pd.Series = delta(df[COLUMN_NAME_CADENCE], np.subtract)
            df[COLUMN_NAME_CADENCE_RATE] = (cadence_delta / df[COLUMN_NAME_DELTA_T].apply(lambda td: td.total_seconds()))

            return df


        trackpoints: dict = self._dict['TrainingCenterDatabase']['Activities']['Activity']['Lap']['Track']
        list_of_trackpoint_dicts = list(trackpoints.values())[0]

        df: pd.DataFrame = pd.DataFrame.from_records(list_of_trackpoint_dicts)
        return prepare_tcx(df)


    @staticmethod
    def read_tcx(file_path: str):
        def read_xml(file_path: str) -> dict:
            project_root_dir = os.path.abspath('.')
            abs_file_path = os.path.join(project_root_dir, file_path)
            with open(abs_file_path, mode='r', encoding='utf-8') as f:
                content = f.read()
                return xml_parse(content)

        # read xml to dict
        return Tcx(read_xml(file_path))
