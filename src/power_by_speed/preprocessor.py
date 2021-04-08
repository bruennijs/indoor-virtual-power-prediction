import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.power_by_speed.regression import COLUMN_NAME_CADENCE, COLUMN_NAME_WATTS
from src.tcx import Tcx
from src.test_data import TrainDataSet

COLUMN_NAME_GEAR_RATIO = 'Cadence-to-power-ratio'

def extract_gear_ratio(X: pd.DataFrame, **kwargs):
    X_ = X.copy()
    X_[COLUMN_NAME_GEAR_RATIO] = X_[COLUMN_NAME_CADENCE] / X_[COLUMN_NAME_WATTS]
    return X_

class ZscoreTransformer(TransformerMixin):

    COLUMN_NAME_GEAR_RATIO_OUTLIER = 'gear-ratio-outlier'
    COLUMN_NAME_GEAR_RATIO_Z_SCORE: str = '{}-z-score'.format(COLUMN_NAME_GEAR_RATIO)

    def __init__(self):
        return

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        return self

    def transform(self, X: pd.DataFrame, **transformparams):
        X_ = X.copy()
        X_[ZscoreTransformer.COLUMN_NAME_GEAR_RATIO_Z_SCORE] = StandardScaler().fit_transform(X_[[COLUMN_NAME_GEAR_RATIO]])
        return X_

def detect_gear_ratio_outlier(X: pd.DataFrame, **kwargs):
    X_ = X.copy()
    outliers: pd.Series = X_[ZscoreTransformer.COLUMN_NAME_GEAR_RATIO_Z_SCORE].apply(lambda v: abs(v) > 1.5)
    X_[ZscoreTransformer.COLUMN_NAME_GEAR_RATIO_OUTLIER] = outliers

    df_outlier_group_count: pd.DataFrame = X_.groupby(by=ZscoreTransformer.COLUMN_NAME_GEAR_RATIO_OUTLIER).count()

    # print(df_outlier_group_count)
    return X_


def drop_gear_ratio_outlier(X: pd.DataFrame, *args, **kwargs):
    X_ = X.copy()
    outliers: pd.DataFrame = X_[X_[ZscoreTransformer.COLUMN_NAME_GEAR_RATIO_OUTLIER]]
    X_t_ = X_.drop(index=outliers.index, axis=0)
    return X_t_


def create_pipeline() -> Pipeline:
    return Pipeline(
        [('gear_ratio_extractor', FunctionTransformer(extract_gear_ratio)),
         ('gear_ratio_z_score', ZscoreTransformer()),
         ('select_gear_ratio_outlier', FunctionTransformer(detect_gear_ratio_outlier)),
         ('drop_gear_ratio_outlier', FunctionTransformer(drop_gear_ratio_outlier))]
    )


