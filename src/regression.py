from math import sqrt

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import max_error, mean_squared_error
from sklearn.linear_model._base import LinearModel

def validate_lin_reg(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series, estimator: LinearModel = LinearRegression) -> LinearModel:
    # train
    estimator.fit(X_train, y_train)

    # test
    y_predicted = estimator.predict(X_test)

    print("Shape X_train/X_test: {}/{}".format(X_train.shape, X_test.shape))
    # Score on X_test
    print("Error R²: {:.2f}".format(estimator.score(X_test, y_test)))
    mse = mean_squared_error(y_test.to_numpy(), y_predicted)
    print("MSE error (mean squared error / variance): {:.2f}".format(mse))
    print("sqrt(MSE) (standard deviation): {:.2f}".format(sqrt(mse)))
    print("Max error: {}".format(max_error(y_test, y_predicted)))
    print("estimator.coefficients: {}".format(estimator.coef_))

    # cross k-fold validation (k=5)
    scores: list = cross_val_score(estimator, X_test.to_numpy(), y_test.to_numpy(), cv=5)
    print("Cross validation: {}".format(scores))
