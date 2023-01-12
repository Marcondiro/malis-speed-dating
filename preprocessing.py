import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import re


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
        DataFrame containing the correspondent labels
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
    y: DataFrame
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
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test, stratify=stratify, random_state=42)

    return X_tr, y_tr, X_te, y_te


def split_train_test_validation(X, y, test=0.1, val=0.1, stratify=True):
    """
    A function splitting a dataset into train, test and validation sets.

    Parameters
    ----------
    X: DataFrame
        The dataset samples
    y: DataFrame
        The corresponding labels
    test: float, optional (default is 0.1)
        The fraction of the dataset we want to keep as test set
    val: float, optional (default is 0.1)
        The fraction of the dataset we want to keep as validation set
    stratify: bool, optional (default is True)
        Whether we want the split to keep the same proportion between classes as the original dataset

    Returns
    -------
    X_tr, y_tr
        Samples and relative labels for the training set
    X_te, y_te
        Samples and relative labels for the test set
    X_val, y_val
        Samples and relative labels for the validation set

    """

    # split it into train and test set
    stratify_ = (y if stratify else None)
    # first, split into train + validation and test set
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test, stratify=stratify_, random_state=42)
    # then, split train + validation into train and validation
    val = val/(1-test)
    stratify_ = (y_tr if stratify else None)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val, stratify=stratify_, random_state=69)

    return X_tr, y_tr, X_te, y_te, X_val, y_val


def grid_search(X, y, estimator, grid, k=5):
    """
    Perform grid search on the provided model/pipe.

    Parameters
    ----------
    X: DataFrame
        The dataset samples
    y: DataFrame
        The corresponding labels
    estimator:
        The sklearn model/pipe we want to apply
    grid: dictionary
        The grid of hyperparameters we want to test for our model
    k: int, optional (default is 5)
        The number of folds for the k-fold validation approach

    Returns
    -------
    dict
        A subset of the GridSearchCV object.
        - best_model - the best estimator found in the grid search
        - best_params - parameters generating such an estimator
        - best_recall - the recall of such an estimator
        - best_balanced_accuracy - the balanced accuracy of such an estimator
        - best_f1 - the f1 of such an estimator
        - all_recall - the mean recall of all estimators generated by the grid
        - all_balanced_accuracy - the mean balanced accuracy of all estimators generated by the grid
        - all_f1 - the mean f1 of all estimators generated by the grid

    """

    # TODO add more? - CHOOSE 1 to get the best
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    metrics = ['balanced_accuracy', 'f1', 'recall']
    best_metric = 'recall'

    # create the GridSearchCV object
    grid_cv = GridSearchCV(estimator, param_grid=grid,
                           cv=k, scoring=metrics, refit=best_metric)
    # run the grid search
    res = grid_cv.fit(X, y)

    out = {
        'best_params': res.best_params_,
        'best_model': res.best_estimator_,
        'best_index': res.best_index_,
        'best_recall': res.cv_results_['mean_test_recall'][res.best_index_],
        'best_balanced_accuracy': res.cv_results_['mean_test_balanced_accuracy'][res.best_index_],
        'best_f1': res.cv_results_['mean_test_f1'][res.best_index_],
        'all_recall': res.cv_results_['mean_test_recall'],
        'all_balanced_accuracy': res.cv_results_['mean_test_balanced_accuracy'],
        'all_f1': res.cv_results_['mean_test_f1']
    }

    return out


def feature_interaction_polynomyal_degreee2(X):
    poly = PolynomialFeatures(2, interaction_only=True)
    return poly.fit_transform(X)


def corresponding_features_interaction(X, drop=True):
    """
    Compute the interactions for the DataFrame X, using the following logic:
        col <- col_x * col_y

    Parameters
    ----------
    X: DataFrame
        The dataset samples
    drop: bool, optional (default is True)
        When True, we drop col_x and col_y, leaving just their interaction col

    Returns
    -------
    DataFrame
        The modified dataset

    """
    # add interaction x-y
    columns = list(X.columns)
    re_x = re.compile(".*_x.*")
    re_y = re.compile(".*_y.*")
    re_sub = re.compile("_x")
    pairs = [[c for c in columns if re_x.match(c)],
             [c for c in columns if re_y.match(c)],
             [re.sub(re_sub, '', c) for c in columns if re_x.match(c)]
             ]
    # copy the dataset
    X_ = X.copy()

    X_[pairs[2]] = np.multiply(X[pairs[0]], np.asarray(X[pairs[1]]))

    # drop the other columns, if necessary
    if drop:
        X_ = X_.drop(pairs[0]+pairs[1], axis='columns', inplace=False)

    return X_


def corresponding_features_custom_interaction(X, drop=True):
    """
    Compute the interactions for the DataFrame X, using the following logic:
    For binary features
    a = b = 1 -> 0
    a = b = 0 -> 1
    a != b    -> 2

    For integer features
    a>6 and b>6 -> abs(a-b)
    else -> 4 + abs(a-b)

    Parameters
    ----------
    X: DataFrame
        The dataset samples
    drop: bool, optional (default is True)
        When True, we drop col_x and col_y, leaving just their interaction col

    Returns
    -------
    DataFrame
        The modified dataset

    """
    # copy the dataset
    data = X.copy()

    cols = ['sports', 'tvsports', 'exercise', 'dining', 'museums',
            'art', 'hiking', 'gaming', 'clubbing',
            'reading', 'tv', 'theater', 'movies',
            'concerts', 'music', 'shopping', 'yoga', ]
    regex = re.compile("(field_cd|race|goal|career_c)_[0-9]*_x")
    cols += [c[:-2] for c in X.columns if regex.match(c)]

    for c in cols:
        data[c] = binary_feature_custom_interaction(X[c + '_x'], X[c + '_y']) if is_binary(X[c + '_x'])\
            else integer_feature_custom_interaction(X[c + '_x'], X[c + '_y'])

        # drop the other columns, if necessary
        if drop:
            data.drop([c + '_x', c + '_y'], axis='columns', inplace=True)

    return data


def is_binary(feature) -> bool:
    return min(feature) in [0, 1] and max(feature) in [0, 1]


def binary_feature_custom_interaction(a: np.array, b: np.array) -> np.array:
    # a = b = 1 -> 0
    # a = b = 0 -> 1
    # a != b    -> 2
    return 1 - np.logical_and(a, b) + np.logical_xor(a, b)


def integer_feature_custom_interaction(a, b) -> np.array:
    result = np.max([a, b], axis=0) - np.min([a, b],
                                             axis=0)  # absolute difference
    # Shared high interest, give more importance (-> penalize non shared and high interest)
    result += 4 - np.logical_and(a > 6, b > 6) * 4
    return result
