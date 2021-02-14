import matplotlib.pyplot as plt
import pandas as pd

def plot2d(X_train: pd.Series, y_train: pd.Series, point_color='black', legend_label='unknown') -> None:
    # Plot X_train -> y_train
    plt.scatter(X_train, y_train,  color=point_color, label=legend_label)
    plt.title("{} = f({})".format(y_train.name, X_train.name))
    plt.xticks()
    plt.yticks()
    plt.legend()
