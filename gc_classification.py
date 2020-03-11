#==================================================================================
#                               GC CLASSIFICATION
#----------------------------------------------------------------------------------
#           Input: Pre-processed gait cycles, Output: LR, SVM & CNN Summaries
#               Classifies according to testing and training data
#==================================================================================
#                                   Imports
#==================================================================================
import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random
from sklearn.preprocessing import LabelEncoder

#==================================================================================
#                                   Constants
#==================================================================================
LABEL = 'id'
BINARY = False
SPLIT_BY_ID = False
TEST_SIZE = 0.5
SEED = random.randint(1, 1000)

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

# Plots confusion matrix without requiring a classifier
def plot_cm(cm, labels, name, cmap, score):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap=cmap)
    plt.xlabel('Predicted ' + LABEL)
    plt.ylabel('True ' + LABEL)
    plt.title(name + ' (Accuracy: {:.2f})'.format(score))
    plt.show()

# Evaluates the keras neural network model w.r.t confusion matrix and ground-truth metrics
def evaluate_nn_summary(model, X_test, y_test, disp_labels):
    pred = model.predict_classes(X_test)
    print('Classification Report CNN:')
    print(metrics.classification_report(y_test, pred))
    cm = metrics.confusion_matrix(y_test, pred)
    print('Confusion matrix for CNN:\n', cm)
    score = sum(y_test == pred) / len(y_test)
    plot_cm(cm, disp_labels, 'CNN', plt.cm.Reds, score)

# Define and Evaluate a Logistic Regression Model
def lr(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'LR', plt.cm.Blues)
    return classifier

# Define and Evaluate a Logistic Regression Model
def svm(X_train, X_test, y_train, y_test):
    classifier = LinearSVC()
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

def splitById(X, y, labels_id):
    id_count = len(np.unique(labels_id))
    test_size = int(TEST_SIZE * id_count)

    test_ids = []
    random.seed(SEED)

    # Ensures that participants in test set are different from training
    # and that the test set has a variety - and doesn't always have the same
    # type of target
    for i in range(0, test_size):
        while True:
            random_id = random.randint(1, id_count-1)
            if(i > 0):
                id_prev = test_ids[-1]
                for j in range(0, len(X)):
                    if(labels_id[j] == id_prev):
                        label_prev = y[j]
                        break
                for j in range(0, len(X)):
                    if(labels_id[j] == random_id):
                        label_curr = y[j]
                        break

                print(random_id, label_curr, test_ids[-1], label_prev)
            if(i == 0 or random_id != test_ids[-1] and label_curr != label_prev):
                test_ids.append(random_id)
                break

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(0, len(X)):
        print(y[i], labels_id[i])

        test = False
        for x in test_ids:
            if(labels_id[i] == x):
                test = True
                break

        if(test):
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])

    train_set = list(zip(X_train, y_train))
    random.shuffle(train_set)
    X_train, y_train = zip(*train_set)

    test_set = list(zip(X_test, y_test))
    random.shuffle(test_set)
    X_test, y_test = zip(*test_set)

    return (X_train, X_test, y_train, y_test)

def mlModels(data_train_test):
    X_train = flattenData(data_train_test[0])
    X_test = flattenData(data_train_test[1])
    y_train = np.array(data_train_test[2])
    y_test = np.array(data_train_test[3])

    #X_train, X_test, y_train, y_test = splitById(X, y, labels_id)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)

    return lr(X_train, X_test, y_train, y_test), svm(X_train, X_test, y_train, y_test)

def nn(data_train_test):
    X_train = data_train_test[0]
    X_test = data_train_test[1]
    y_train = data_train_test[2]
    y_test = data_train_test[3]

    # Pre-process to (samples, time-steps, features)
    X_train = np.array([np.array(x).transpose() for x in X_train])
    X_test = np.array([np.array(x).transpose() for x in X_test])

    y = [*y_train, *y_test]
    label_encoder = LabelEncoder()
    disp_labels = sorted(list(dict.fromkeys(y).keys()))
    y_test = label_encoder.fit_transform(y_test)
    y_train = label_encoder.fit_transform(y_train)
    y = [*y_train, *y_test]
    n_outputs = len(np.unique(y))
    y_train = keras.utils.to_categorical(y_train, num_classes=n_outputs)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=SEED)
    #y_train = keras.utils.to_categorical(y_train, num_classes=n_outputs)

    filter1 = 101
    filter2 = 162
    kernel = 10
    dropout_rate = 0.5

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

    # https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
    model_m = Sequential()
    model_m.add(Conv1D(filter1, kernel, activation='relu', input_shape=(n_timesteps, n_features)))
    model_m.add(Conv1D(filter1, kernel, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(filter2, kernel, activation='relu'))
    model_m.add(Conv1D(filter2, kernel, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(dropout_rate))
    model_m.add(Dense(n_outputs, activation='softmax'))
    #print(model_m.summary())
    if(BINARY): loss = 'binary_crossentropy'
    else: loss = 'categorical_crossentropy'
    model_m.compile(loss=loss,
                    optimizer='adam', metrics=['accuracy'])

    epochs = 50 # 50
    #TODO: Might consider batch and CV, after part sepera

    model_m.fit(X_train, y_train, epochs=epochs, verbose=1, validation_split=TEST_SIZE)
    evaluate_nn_summary(model_m, X_test, y_test, disp_labels)
    return model_m

#==================================================================================
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data.pickle', 'rb') as f:
    data = pickle.load(f)
with open('..\\classifier_data\\labels_' + LABEL + '.pickle', 'rb') as f:
    labels = pickle.load(f)
with open('..\\classifier_data\\labels_id.pickle', 'rb') as f:
    labels_id = pickle.load(f)

if(SPLIT_BY_ID):
    data_train_test = splitById(data, labels, labels_id)
else:
    data_train_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, random_state=SEED)

mlModels(data_train_test)
#nn(data_train_test) # TODO: Fix CNN bug: inconsistencies in confusion matrix
print('SEED:', SEED)
