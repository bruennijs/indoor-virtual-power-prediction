import unittest
from functools import reduce

import pandas as pd
from pandas import DataFrame, merge
from sklearn.model_selection import cross_val_score, train_test_split, KFold

from src.power_by_speed.regression import create_pipeline
from src.tcx import Tcx, COLUMN_NAME_CADENCE, COLUMN_NAME_WATTS, COLUMN_NAME_SPEED
from src.test_data import TrainDataSet


class RegressionByPowerTest(unittest.TestCase):

    df_tacx: DataFrame
    df_app: DataFrame

    #@classmethod
    #def setUpClass(cls):

    def setUp(self):
        self.df_app: DataFrame = TrainDataSet(Tcx.read_tcx(file_path='./../tcx/cadence_1612535177298-gear7.tcx')).get_dataframe()
        self.df_tacx: DataFrame = TrainDataSet(Tcx.read_tcx(file_path='./../tcx/tacx-activity_6225123072-gear7-resistance3.tcx')).get_dataframe()
        self.pipeline = create_pipeline('./../tcx/cadence_1612535177298-gear7.tcx')

    def test_power_prediction(self):

        df_joined: DataFrame = pd.merge(self.df_tacx, self.df_app, on=COLUMN_NAME_CADENCE, how='inner', suffixes=('_tacx', '_app'))
        self.assertIn('{}_app'.format(COLUMN_NAME_SPEED), df_joined.columns)
        self.assertIn('{}_tacx'.format(COLUMN_NAME_WATTS), df_joined.columns)

    def test_score(self):
        # cross_val_score()
        X_train, X_test, y_train, y_test = train_test_split(self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1), self.df_tacx[[COLUMN_NAME_WATTS]], train_size=0.95)

        self.assertGreater(self.pipeline.fit(X_train, y_train).score(X_test, y_test), 0.98)

    def test_cv(self):
        scores: list[float] = cross_val_score(self.pipeline,
                                              X=self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1),
                                              y=self.df_tacx[[COLUMN_NAME_WATTS]],
                                              cv=KFold(n_splits=5, shuffle=True, random_state=23628763))

        # THEN
        self.assertTrue(all(scores > 0.98), 'All k score results greater 0.98')


if __name__ == '__main__':
    unittest.main()
