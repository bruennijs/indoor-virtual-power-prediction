import matplotlib.pyplot as plt
import pandas as pd

def plot2d(X: pd.Series, y: pd.Series, point_color='black', legend_label='unknown') -> None:
    # Plot X_train -> y_train
    plt.scatter(X, y, color=point_color, label=legend_label)
    plt.title("{} = f({})".format(y.name, X.name))
    plt.xticks()
    plt.yticks()
    plt.legend()

def histogram(x, bins: int=25, title='unknown'):
    plt.hist(x, bins, density=False, facecolor='g', alpha=0.75)
    plt.title("Density of {}".format(title))

def show():
    plt.show()
