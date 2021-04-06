import unittest
from functools import reduce

import pandas as pd
from pandas import DataFrame, merge

import numpy as np

from sklearn.metrics import make_scorer, max_error, mean_absolute_percentage_error
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
        X_train, X_test, y_train, y_test = train_test_split(self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1), self.df_tacx[[COLUMN_NAME_WATTS]], train_size=0.5)

        self.assertGreater(self.pipeline.fit(X_train, y_train).score(X_test, y_test), 0.99)

    def test_manual_prediction(self):
        # cross_val_score()
        X_train, X_test, y_train, y_test = train_test_split(self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1), self.df_tacx[[COLUMN_NAME_WATTS]], train_size=0.5)

        self.pipeline.fit(X_train, y_train)

        # assert in expected power range
        y_pred = self.pipeline['estimator'].predict(pd.DataFrame(data=[24.1], index=[5]))
        self.assertTrue(y_pred[0] > 111.0, msg="110 < y")
        self.assertTrue(y_pred[0] < 113.0, msg="111 > y")

    def test_cv_scorer_r2(self):
        scores: list[float] = cross_val_score(self.pipeline,
                                              X=self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1),
                                              y=self.df_tacx[[COLUMN_NAME_WATTS]],
                                              cv=KFold(n_splits=5, shuffle=True, random_state=37648))

        # THEN
        self.assertTrue(all(scores > 0.98), 'All k score results greater 0.98')

    def test_cv_scorer_max_error(self):
        scores: list[float] = cross_val_score(self.pipeline,
                                              X=self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1),
                                              y=self.df_tacx[[COLUMN_NAME_WATTS]],
                                              cv=KFold(n_splits=5, shuffle=True, random_state=37648),
                                              scoring=make_scorer(max_error))

        # THEN
        self.assertLess(max(scores), 1.0, 'All k max errors less than 1.0 watts')

    def test_cv_scorer_max_abs_percentage_error(self):
        scores: list[float] = cross_val_score(self.pipeline,
                                              X=self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1),
                                              y=self.df_tacx[[COLUMN_NAME_WATTS]],
                                              cv=KFold(n_splits=5, shuffle=True, random_state=37648),
                                              scoring=make_scorer(mean_absolute_percentage_error))

        # THEN
        self.assertLess(max(scores), 0.01, 'All mean absolute percentage error less than 2.0 %')

    def test_cv_scorer_percentile_abs_error(self):
        scores: list[float] = cross_val_score(self.pipeline,
                                              X=self.df_tacx.drop(COLUMN_NAME_WATTS, axis=1),
                                              y=self.df_tacx[[COLUMN_NAME_WATTS]],
                                              cv=KFold(n_splits=5, shuffle=True, random_state=37648),
                                              scoring=make_scorer(self.percentile_absolute_error))

        # THEN
        self.assertLess(max(scores), 0.01, '99 percent of all abs errors is less than 3 %')


    def max_error(self, y_t: pd.DataFrame, y_predicted):
        y_diff = y_t - y_predicted
        return abs(max(y_diff.iloc[:, 0]))


    def percentile_absolute_error(self, y_t: pd.DataFrame, y_predicted, quantile: float = 99, **kwargs):
        y_diff = (y_t - y_predicted)
        abs_errors: pd.Series = y_diff.iloc[:, 0]
        abs_errors: pd.Series = abs_errors / max(y_t.iloc[:, 0])
        percentile = np.percentile(abs_errors.to_numpy(), quantile)
        return percentile


if __name__ == '__main__':
    unittest.main()
