import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.tcx import Tcx
from src.test_data import TrainDataSet

COLUMN_NAME_SPEED = 'Speed'
COLUMN_NAME_CADENCE = 'Cadence'
COLUMN_NAME_WATTS = 'Ext.Watts'
COLUMN_NAME_SPEED_APP = 'Speed-app'


def select_features(X: pd.DataFrame, **kwargs):
    return X[[COLUMN_NAME_SPEED_APP]]

def print_debug(X, **kwargs):
    print(X)
    return X

class JoinAppSpeedTransformer(TransformerMixin):

    def __init__(self, filename: str):
        self.df_app = TrainDataSet(Tcx.read_tcx(filename)).get_dataframe()
        self.estimator = self._train_regressor(self.df_app[[COLUMN_NAME_CADENCE]], self.df_app[COLUMN_NAME_SPEED])

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame, **transformparams):

        y = self.estimator.predict(X[[COLUMN_NAME_CADENCE]])

        X_ = X.copy()
        X_[COLUMN_NAME_SPEED_APP] = pd.Series(y, index=X.index)
        return X_

    def _create_regressor(self):
        return KNeighborsRegressor(n_neighbors=3, weights='uniform')

    def _train_regressor(self, X, y):
        return self._create_regressor().fit(X, y)

def create_pipeline(tcx_app_filename: str) -> Pipeline:
    return Pipeline(
        [('app_speed_extractor', JoinAppSpeedTransformer(tcx_app_filename)),
         ('feature_selector', FunctionTransformer(select_features)),
         #('printer', FunctionTransformer(print_debug)),
         ('estimator', LinearRegression())]
    )

