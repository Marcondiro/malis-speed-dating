##########################################
#                                        #
#                PIPELINE                #
#                                        #
##########################################

from preprocessing import load_dataset, split_dataset
from sklearn.model_selection import GridSearchCV

import models as models

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
    metrics = ['balanced_accuracy', 'f1']
    best_metric = 'f1'

    # create the GridSearchCV object
    grid_cv = GridSearchCV(model, param_grid=grid, cv=k, scoring=metrics, refit=best_metric)
    # run the grid search
    out = grid_cv.fit(X, y)

    # TODO see what happens?
    a = 1

    return






##########################################
#                                        #
#               PARAMETERS               #
#                                        #
##########################################
test_size = 0.3         # the ratio of the dataset we want to use as test set
stratify = True         # Whether we want the split to keep the same proportion between classes as the original dataset
k = 10                  # The number of folds for the stratified k fold 

model_f = models.svm    # The model we want to use




if __name__ == "__main__":

    # first, we load the dataset
    X, y = load_dataset('./data/data.pkl')
    # then, we split it
    X_tr, y_tr, X_te, y_te = split_dataset(X, y, test=test_size, stratify=stratify)
    # get the grid and the model
    model, grid = model_f()
    # call grid_search
    grid_search(X_tr, y_tr, model, grid, k)

    