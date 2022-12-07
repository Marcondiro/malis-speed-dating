##########################################
#                                        #
#           DATA PREPROCESSING           #
#                                        #
##########################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
    X.drop('match', inplace=True)

    return X, y



def split_dataset(X, y, test=0.3, stratify=True, k=5):
    """
    Splits the dataset into training and test set, then create KFold model.

    
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
    k: int, optional (default is 5)
        The number of folds for the k-fold validation approach

    Returns
    -------
    X_tr, y_tr
        Samples and relative labels for the training set
    X_te, y_te
        Samples and relative labels for the test set
    kf
        KFold model (to be applied on X_tr, y_tr)

    """
    
    # split it into train and test set
    stratify = (y if stratify else None)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test, stratify=stratify)

    # create the k-fold class
    kf = KFold(n_splits=k, shuffle=True)

    return X_tr, y_tr, X_te, y_te, kf



#########################################
#                                       #
#             USAGE EXAMPLE             #
#                                       #
#########################################    

if __name__ == "__main__":

    # first, we load the dataset
    X, y = load_dataset('./data/data.pkl')
    # then, we split it
    X_tr, y_tr, X_te, y_te, kf = split_dataset(X, y)





