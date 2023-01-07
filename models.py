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
# to perform logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # to perform RandomForest
from sklearn.svm import SVC  # to perform SVM classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


def knn():
    K_MIN = 1
    K_MAX = 10

    knn = KNeighborsClassifier()
    grid = {
        'n_neighbors': list(range(K_MIN, K_MAX+1))
    }

    return knn, grid


def logistic_regression():
    MAX_ITER = 250000000  # maximum number of iterations

    # C = inverse of regularization strength (smaller = stronger regularization)
    lr = LogisticRegression(solver='newton-cg', max_iter=MAX_ITER, class_weight='balanced')
    grid = {
        'C': [10, 100, 10000]
    }

    return lr, grid


def svm():
    MAX_ITER = 25000  # maximum number of iterations

    clf = SVC(max_iter=MAX_ITER, class_weight='balanced')
    grid = {
        'kernel': ['linear', 'rbf']
    }

    return clf, grid


def random_forest():

    rf = RandomForestClassifier(max_features="sqrt", bootstrap=True, oob_score=True, class_weight='balanced')
    """
    grid = {
        'n_estimators': [200, 500], 
        'max_features': ['auto', 'sqrt', 'log2'], 
        'max_depth': [4, 5, 6, 7, 8], 
        'criterion': ['gini', 'entropy']
    }
    """
    grid = {
        'min_samples_split': [2, 5],
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy']
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


def over_sampling(estimator_f, **params):

    estimator, grid = estimator_f(**params)
    smote = SMOTE()
    # if estimator is a pipeline
    if estimator.__class__.__name__ == 'Pipeline':
        pipe = estimator
        pipe.steps.insert(0, ['over_sampler', smote])
    else:
        # create the pipeline
        pipe = Pipeline(steps=[('over_sampler', smote), ('model', estimator)])
        # update the grid
        grid = {f"model__{key}": val for key, val in grid.items()}

    return pipe, grid
