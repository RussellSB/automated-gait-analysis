#==================================================================================
#                               GC CLASSIFIERS
#----------------------------------------------------------------------------------
#               Input: Pre-processed gait cycles, Output: Classification
#               Classifies according to testing and training data
#==================================================================================
#                                   Imports
#==================================================================================
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Dropout
# from keras.layers.convolutional import Conv1D, MaxPooling1D

#==================================================================================
#                                   Constants
#==================================================================================
LABEL = 'age'

#==================================================================================
#                                   Methods
#==================================================================================

# Evaluates the sklearn model w.r.t confusion matrix and ground-truth metrics
def evaluate_sk_summary(classifier, X_test, y_test, sklearnType, cmap):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    predicted = classifier.predict(X_test)
    print("Classification report for", sklearnType, "classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test,
                                         include_values=True, cmap=cmap)
    disp.figure_.suptitle("")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    plt.xlabel('Predicted ' + LABEL)
    plt.ylabel('True ' + LABEL)
    score = classifier.score(X_test, y_test)
    plt.title(sklearnType + ' (Accuracy: {:.2f})'.format(score))
    plt.show()
    plt.close()

# Define and Evaluate a Logistic Regression Model
def lr(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'LR', plt.cm.Blues)
    return classifier

# Define and Evaluate a Logistic Regression Model
def svm(X_train, X_test, y_train, y_test):
    from sklearn import svm
    classifier = svm.LinearSVC()
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'SVM', plt.cm.Greens)
    return classifier

def convertDataToOneVector(data):
    X = []
    for d in data:
        x = []
        for i in range(0, 8):
            x.extend(d[i].tolist())
        X.append(x)
    X = np.array(X)
    return X

def mlModels(data, labels):
    X = convertDataToOneVector(data)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
    return lr(X_train, X_test, y_train, y_test), svm(X_train, X_test, y_train, y_test)

def nn():
    X = np.array([np.array(x).transpose() for x in data])  # (samples, time-steps, features)
    verbose = 1
    epochs = 20
    batch_size = 64
    n_outputs = len(np.unique(labels))
    n_timesteps, n_features = X.shape[1], X.shape[2]

#==================================================================================
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data.pickle', 'rb') as f:
    data = pickle.load(f)
with open('..\\classifier_data\\labels_' + LABEL + '.pickle', 'rb') as f:
    labels = pickle.load(f)

mlModels(data, labels)
