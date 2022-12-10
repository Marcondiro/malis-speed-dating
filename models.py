##########################################
#                                        #
#                 MODELS                 #
#                                        #
##########################################

"""
All methods in the models.py file return a classifier and a grid dictionary.

# FIXME - configure all grids
"""

from sklearn.neighbors import KNeighborsClassifier  # to perform kNN
from sklearn.linear_model import LogisticRegression # to perform logistic regression
from sklearn.ensemble import RandomForestClassifier # to perform RandomForest 
from sklearn.svm import SVC  # to perform SVM classification

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
        'C': [100]
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