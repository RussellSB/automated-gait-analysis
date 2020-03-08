#==================================================================================
#                               GC CLASSIFIERS
#----------------------------------------------------------------------------------
#               Input: Pre-processed gait cycles, Output: Classification
#               Classifies according to testing and training data
#==================================================================================
#                                   Imports
#==================================================================================
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

#==================================================================================
#                                   Constants
#==================================================================================
LABEL = 'gender'
TEST_SIZE = 0.2

#==================================================================================
#                                   Methods
#==================================================================================

# Evaluates the sklearn model w.r.t confusion matrix and ground-truth metrics
def evaluate_sk_summary(classifier, X_test, y_test, sklearnType, cmap):
    predicted = classifier.predict(X_test)
    print("Classification report for", sklearnType, "classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test,
                                         include_values=True, cmap=cmap)
    disp.figure_.suptitle("")
    print("Confusion matrix for " + sklearnType + ":\n%s" % disp.confusion_matrix)

    plt.xlabel('Predicted ' + LABEL)
    plt.ylabel('True ' + LABEL)
    score = classifier.score(X_test, y_test)
    plt.title(sklearnType + ' (Accuracy: {:.2f})'.format(score))
    plt.show()
    plt.close()

# Evaluates the keras neural network model w.r.t confusion matrix and ground-truth metrics
def evaluate_nn_summary(model, X_test, y_test, batch_size):
    predictions = model.predict(X_test)
    pred = []
    for p in predictions:
        p = 1 if p[1] > p[0] else 0
        pred.append(p)
    print('Classification Report NN:')
    print(metrics.classification_report(y_test, pred))

    cm = metrics.confusion_matrix(y_test, pred)
    print('Confusion matrix for CNN:\n', cm)
    plt.imshow(cm, cmap=plt.cm.Reds)

    lab = np.sort(np.unique(pred))

    plt.xticks(lab)
    plt.yticks(lab)
    plt.ylim(1.5, -0.5)

    plt.xlabel('Predicted ' + LABEL)
    plt.ylabel('True ' + LABEL)
    _, score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    plt.title('CNN (Accuracy: {:.2f})'.format(score))
    plt.colorbar()
    plt.show()

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

# Flattens each 8x101 sample to one single long 1x808 vector
def flattenData(data):
    X = []
    for d in data:
        x = []
        for i in range(0, 8):
            x.extend(d[i].tolist())
        X.append(x)
    X = np.array(X)
    return X

def mlModels(data, labels):
    X = flattenData(data)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return lr(X_train, X_test, y_train, y_test), svm(X_train, X_test, y_train, y_test)

def nn(data, labels):
    X = np.array([np.array(x).transpose() for x in data])  # (samples, time-steps, features)
    # y = np.array([y-1 for y in labels])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    epochs = 50
    batch_size = 400
    n_outputs = len(np.unique(labels))
    n_timesteps, n_features = X.shape[1], X.shape[2]

    model_m = Sequential()
    model_m.add(Conv1D(100, 10, activation='relu', input_shape=(n_timesteps, n_features)))
    model_m.add(Conv1D(100, 10, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(Conv1D(160, 10, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(n_outputs, activation='softmax'))  # 1 should be n_outputs
    print(model_m.summary())

    model_m.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    history = model_m.fit(X_train, y_train,
                          batch_size=batch_size, epochs=epochs,
                          verbose=1)  # validation_split = 0.2




    #evaluate_nn_summary(model_m, X_test, y_test, batch_size)

    return model_m

#==================================================================================
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data.pickle', 'rb') as f:
    data = pickle.load(f)
with open('..\\classifier_data\\labels_' + LABEL + '.pickle', 'rb') as f:
    labels = pickle.load(f)

#mlModels(data, labels)
nn(data, labels)
