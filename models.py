##########################################
#                                        #
#                 MODELS                 #
#                                        #
##########################################

"""
All methods in the models.py file return a classifier and a grid dictionary.

# FIXME - configure all grids
# TODO - add the scaling feat.?
"""

from sklearn.neighbors import KNeighborsClassifier  # to perform kNN
from sklearn.linear_model import LogisticRegression # to perform logistic regression
from sklearn.ensemble import RandomForestClassifier # to perform RandomForest 
from sklearn.svm import SVC  # to perform SVM classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def knn():
    k_min = 1
    k_max = 10

    knn = KNeighborsClassifier()
    grid = {
        'n_neighbors': list(range(k_min, k_max+1))
    }

    return knn, grid


def logistic_regression():
    max_iter = 250000  # maximum number of iterations
    
    # C = inverse of regularization strength (smaller = stronger regularization)
    lr = LogisticRegression(solver='newton-cg', max_iter=max_iter)
    grid = {
        'C': [10, 100, 10000]
    }

    return lr, grid


def svm():
    max_iter = 25000  # maximum number of iterations
    
    clf = SVC(max_iter=max_iter)
    grid = {
        'kernel': ['linear', 'rbf']
    }

    return clf, grid


def random_forest():

    rf = RandomForestClassifier()
    grid = {
        'min_samples_split': [2, 5]
    }

    return rf, grid


#########################################
#                                       #
#               PIPELINES               #
#                                       #
#########################################


def scaling(model_f):
    """
    Generates a Pipeline with scaling and a given model.

    Parameters
    ----------
    model_f: function
        The 'models' function correspondent to the model we want to apply

    Returns
    -------
    pipe
        The pipeline we will apply the grid search on
    grid
        The grid we will use for the grid search
    """
    # get all pipeline stages
    scaler = StandardScaler()
    model, grid = model_f()

    # create the pipeline
    pipe = Pipeline(steps=[('scaler', scaler), ('model', model)])

    # update the grid
    grid = {f"model__{key}": val for key, val in grid.items()}

    return pipe, grid


def pca(n_components, model_f):
    """
    Generates a Pipeline with scaling, PCA and a given model.

    Parameters
    ----------
    n_components: list
        The list of values we want to try for the PCA
    model_f: function
        The 'models' function correspondent to the model we want to apply

    Returns
    -------
    pipe
        The pipeline we will apply the grid search on
    grid
        The grid we will use for the grid search
    """
    # get all pipeline stages
    pca = PCA()
    scaler = StandardScaler()
    model, grid = model_f()

    # create the pipeline
    pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('model', model)])

    # update the grid
    grid = {f"model__{key}": val for key, val in grid.items()}
    grid['pca__n_components'] = n_components

    return pipe, grid
