##########################################
#                                        #
#           DATA PREPROCESSING           #
#                                        #
##########################################

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessing(dataset_file, split):
    """
    Prepares the dataset, applying different kinds of data preprocessing.

    
    Parameters
    ----------
    dataset_file: str
        The name of the file containing the dataset (.pkl)
    split: dict
        The set of parameters regarding the dataset split.
        It should contain the following keys:
        - 'test': float - the fraction of the dataset we want to keep as test set
        - 'stratify': bool - whether we want the split to keep the same proportion between classes as
           the original dataset
        - 'k': int - the number of folds for the k-fold validation approach




    # FIXME - just some template to follow for later
    print_cols : bool, optional
        A flag used to print the columns to the console (default is False)

    Returns
    -------
    list
        a list of strings representing the header columns



    """
    # first, load the dataframe
    X = pd.read_pickle(dataset_file)

    # extract the label column, and remove it from the original dataframe
    y = X['match']
    X.drop('match', inplace=True)
    
    # split it into train and test set
    stratify = (y if split.stratify else None)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split.test, stratify=stratify)



    






split = {
    'test': 0.3,
    'stratify': True,
    'k': 10
}


preprocessing('data/data.pkl')

