##########################################
#                                        #
#           DATA PREPROCESSING           #
#                                        #
##########################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import models


def load_dataset(dataset_file):
    """
    Reads the dataset from file, extracting the label column.

    
    Parameters
    ----------
    dataset_file: str
        The name of the file containing the dataset (.pkl)

    Returns
    -------
    X
        DataFrame containing the dataset samples
    y
        Column (FIXME?) containing the correspondent labels
    """

    # first, load the dataframe
    X = pd.read_pickle(dataset_file)

    # extract the label column, and remove it from the original dataframe
    y = X['match']
    X.drop('match', axis='columns', inplace=True)

    return X, y



def split_dataset(X, y, test=0.3, stratify=True):
    """
    An utility wrapper for train_test_split.

    Parameters
    ----------
    X: DataFrame
        The dataset samples
    y: FIXME
        The corresponding labels
    test: float, optional (default is 0.3)
        The fraction of the dataset we want to keep as test set
    stratify: bool, optional (default is True)
        Whether we want the split to keep the same proportion between classes as the original dataset

    Returns
    -------
    X_tr, y_tr
        Samples and relative labels for the training set
    X_te, y_te
        Samples and relative labels for the test set

    """
    
    # split it into train and test set
    stratify = (y if stratify else None)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test, stratify=stratify)

    return X_tr, y_tr, X_te, y_te



def grid_search(X, y, model, grid, k=5):

    """
    Perform grid search on the provided model.
        Parameters
    ----------
    X: DataFrame
        The dataset samples
    y: DataFrame
        The corresponding labels
    model:
        The sklearn model we want to apply
    grid: dictionary
        The grid of hyperparameters we want to test for our model 
    k: int, optional (default is 5)
        The number of folds for the k-fold validation approach

    Returns
    -------
    # FIXME
    
    """

    # TODO add more? - CHOOSE 1 to get the best
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    metrics = ['balanced_accuracy', 'f1', 'recall']
    best_metric = 'recall'

    # create the GridSearchCV object
    grid_cv = GridSearchCV(model, param_grid=grid, cv=k, scoring=metrics, refit=best_metric)
    # run the grid search
    out = grid_cv.fit(X, y)

    return out



##########################################
#                                        #
#               PARAMETERS               #
#                                        #
##########################################
test_size = 0.2         # the ratio of the dataset we want to use as test set
stratify = True         # Whether we want the split to keep the same proportion between classes as the original dataset
k = 5                  # The number of folds for the stratified k fold 

model_f = models.logistic_regression    # The model we want to use


if __name__ == "__main__":

    # first, we load the dataset
    X, y = load_dataset('./data/data.pkl')
        # first, we load the dataset
    X, y = load_dataset('./data/data.pkl')
    # then, we split it
    X_tr, y_tr, X_te, y_te = split_dataset(X, y, test=test_size, stratify=stratify)
    # get the grid and the model
    model, grid = model_f()
    # call grid_search
    grid_search(X_tr, y_tr, model, grid, k)
