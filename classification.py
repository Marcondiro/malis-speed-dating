import sklearn.datasets # for the test dataset
from sklearn.neighbors import KNeighborsClassifier  # to perform kNN
from sklearn import metrics  # to check accuracy for kNN
import matplotlib.pyplot as plt  # for the final kNN plot
from sklearn.linear_model import LogisticRegression  # to perform logistic regression
from sklearn import svm  # to perform SVM classification
from sklearn.model_selection import train_test_split  # TODO: remove


#TODO: remove
def generate_sets(data):
    data_train, data_test = train_test_split(data, test_size=0.2, stratify=data["labels"])
    X_train = data_train.drop("labels", axis=1)
    y_train = data_train["labels"]
    X_test = data_test.drop("labels", axis=1)
    y_test = data_test["labels"]
    
    return X_train, y_train, X_test, y_test

#TODO: choose appropriate k_min and k_max
def knn(X_train, y_train, X_test, y_test):
    k_min = 1
    k_max = 10
    k_best = 0
    accuracy_best = 0
    k_range = range(k_min, k_max + 1)
    accuracy_list = []
    
    # Test kNN with varying k
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)  # Create a classifier object
        knn.fit(X_train, y_train)  # Train the model
        y_pred = knn.predict(X_test)  # Perform the testing
        accuracy = metrics.accuracy_score(y_test, y_pred)  # Compute the accuracy
        accuracy_list.append(accuracy)  # store the accuracy for every k
        
        # Store k with the best accuracy
        if k == k_min:
            k_best = k
            accuracy_best = accuracy
        elif accuracy > accuracy_best:
            k_best = k
            accuracy_best = accuracy
            
    print("kNN:")
    
    # Plot the accuracy for every k in the range
    plt.figure("kNN")
    plt.plot(k_range, accuracy_list)
    plt.xlabel('Values of \'k\'')
    plt.ylabel('Testing accuracy')
    plt.ylim(top=1)
    plt.show()
    
    print("Best value of k: " + str(k_best) + " (range: [" + str(k_min) + ", " + str(k_max) + "])")
    print("Accuracy: " + str(accuracy_best))

def logistic_regression(X_train, y_train, X_test, y_test):
    max_iter = 250  # maximum number of iterations
    
    # Model training
    # C = inverse of regularization strength (smaller = stronger regularization)
    logistic_model = LogisticRegression(C=1000000, solver='newton-cg', max_iter=max_iter).fit(X_train,y_train)
    
    print("\n\nLogistic regression:")
    
    # Print the coefficient
    print('Model coefficients:')
    print(logistic_model.coef_)
    print('Accuracy: ' + str(logistic_model.score(X_test,y_test)))
    print('Actual number of iterations: ' + str(logistic_model.n_iter_) + " (max: " + str(max_iter) + ")")

#TODO: choose appropriate kernel(s)
def svm(X_train, y_train, X_test, y_test):
    max_iter = 250  # maximum number of iterations
    kernel = 'linear'  # which kernel function to use
    
    # Model training
    clf = sklearn.svm.SVC(kernel=kernel, max_iter=max_iter).fit(X_train, y_train) # Linear Kernel

    # Predict the output labels
    y_pred = clf.predict(X_test)
    
    print('\n\nSVM:')
    print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
    print('Support vectors:\n' + str(clf.support_vectors_))


if __name__ == "__main__":
    # Generate a dataset - iris flower classification
    data, labels = sklearn.datasets.load_iris(return_X_y=True, as_frame=True)  # 'as_frame=True' to receive a DataFrame
    data["labels"] = labels
    #print(data)

    # Generate training and testing sets
    X_train, y_train, X_test, y_test = generate_sets(data)
    
    # Run the algorithms
    knn(X_train, y_train, X_test, y_test)
    logistic_regression(X_train, y_train, X_test, y_test)
    svm(X_train, y_train, X_test, y_test)
