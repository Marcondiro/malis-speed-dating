#########################################
#                                       #
#               UTILITIES               #
#                                       #
#########################################

import pandas as pd
import matplotlib.pyplot as plt


def print_full(x):
    """
    Prints object x with no truncation.
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def plot_grid_results(grid_out, model_name):

    fig = plt.figure()
    fig.set_size_inches(12.5, 4)
    fig.suptitle(f"[{model_name}] - Comparison of classification results")

    x = range(1, len(grid_out['all_recall'])+1)

    # recall
    plt.subplot(1, 3, 1)
    plt.title("Recall")
    plt.plot(x, grid_out['all_recall'])
    plt.xlabel('attempt #')
    plt.ylabel('Recall (mean)')

    # balanced accuracy
    plt.subplot(1, 3, 2)
    plt.title("Balanced Accuracy")
    plt.plot(x, grid_out['all_balanced_accuracy'])
    plt.xlabel('attempt #')
    plt.ylabel('Balanced Accuracy (mean)')

    # f1
    plt.subplot(1, 3, 3)
    plt.title("f1")
    plt.plot(x, grid_out['all_f1'])
    plt.xlabel('attempt #')
    plt.ylabel('f1 (mean)')

    fig.tight_layout()
    plt.show()


def print_log(out, is_grid=True):
    is_grid and print("Best model (according to recall): " + str(out['best_params']))
    print('-----------------------------------------')
    print("Recall : " + str(out['best_recall']))
    print("Balanced accuracy: " + str(out['best_balanced_accuracy']))
    print("f1: " + str(out['best_f1']))