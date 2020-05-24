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
from sklearn.dummy import DummyClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import random
from sklearn.preprocessing import LabelEncoder
from statistics import mean, stdev
from tqdm import trange

#==================================================================================
#                                   Constants
#==================================================================================
# General hyper-parameters
LABEL = 'abnormality' # options: 'abnormality', 'gender', 'age', 'id'
BINARY = True
SPLIT_BY_ID = True
SHOW_TEST_IDs = False
TEST_SIZE = 0.2
REPEAT = True
REPEAT_AMOUNT = 50
SEED = random.randint(1, 1000)

if(REPEAT):
    acc_dc = []
    acc_lr = []
    acc_svm = []
    acc_cnn = []

# Logistic regression hyper-parameter
solver = 'liblinear' # for small datasets, improves performance

# CNN hyper-parameters
epochs = 100
filter1 = 101
filter2 = 162
kernel = 10
dropout_rate = 0.5

# more epochs for id than others
# NOTEWORTHY SEEDS
### V. GOOD
# id: 182, 495 (81%, 81%, 72%), 179, 988 # identification needs more epochs and test size
# gender: 238 (91%, 94%, 66%)
# abnormality: 168 (97%, 97%, 100%)
# abnormality: 899 (70%, 70%, 74% - all of whats in test set is normal)
# abnormality: 577 (98%, 98%, 73%), 588

### V. BAD
# abnormality: 854, 791, 267, 617, 476, 609, 83, 482, 820, 112... etc (CNN classifies everything as normal -
#                                               except SVM and LR classify at 70%)
# age: all.... (around 30%)

#==================================================================================
#                                   Methods
#==================================================================================
# Evaluates the sklearn model w.r.t confusion matrix and ground-truth metrics
def evaluate_sk_summary(classifier, X_test, y_test, sklearnType, cmap):
    score = classifier.score(X_test, y_test)
    if(REPEAT):
        if(sklearnType == 'dc'): acc_dc.append(score)
        if (sklearnType == 'lr'): acc_lr.append(score)
        if (sklearnType == 'svm'): acc_svm.append(score)

    if(not REPEAT):
        predicted = classifier.predict(X_test)
        print("Classification report for", sklearnType, "classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(y_test, predicted)))
        disp = metrics.plot_confusion_matrix(classifier, X_test, y_test,
                                             include_values=True, cmap=cmap)
        disp.figure_.suptitle("")
        print("Confusion matrix for " + sklearnType + ":\n%s" % disp.confusion_matrix)
        plt.xlabel('Predicted ' + LABEL)
        plt.ylabel('True ' + LABEL)
        plt.title('Confusion Matrix (Accuracy: {:.2f})'.format(score))
        plt.show()
        plt.close()

# Plots confusion matrix without requiring a classifier
def plot_cm(cm, labels, name, cmap, score):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap=cmap)
    plt.xlabel('Predicted ' + LABEL)
    plt.ylabel('True ' + LABEL)
    plt.title('Confusion Matrix (Accuracy: {:.2f})'.format(score)) # name
    plt.show()

# Evaluates the keras neural network model w.r.t confusion matrix and ground-truth metrics
def evaluate_nn_summary(model, X_test, y_test, disp_labels):
    pred = model.predict_classes(X_test)
    score = sum(y_test == pred) / len(y_test)

    if(REPEAT):
        acc_cnn.append(score)

    if(not REPEAT):
        print('Classification Report CNN:')
        print(metrics.classification_report(y_test, pred))
        cm = metrics.confusion_matrix(y_test, pred)
        print('Confusion matrix for CNN:\n', cm)
        plot_cm(cm, disp_labels, 'CNN', plt.cm.Reds, score)

# Define and Evaluate a baseline model (Dummy classifier)
def dc(X_train, X_test, y_train, y_test):
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'dc', plt.cm.Reds)
    return classifier

# Define and Evaluate a Logistic Regression Model
def lr(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(solver=solver, random_state=SEED)
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'lr', plt.cm.Reds)
    return classifier

# Define and Evaluate an SVM Model
def svm(X_train, X_test, y_train, y_test):
    classifier = LinearSVC(random_state=SEED)
    classifier.fit(X_train, y_train)
    evaluate_sk_summary(classifier, X_test, y_test, 'svm', plt.cm.Reds)
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
            if(i == 0 or random_id != test_ids[-1] and label_curr != label_prev):
                test_ids.append(random_id)
                break

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(0, len(X)):
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

    if(not REPEAT):
        print('==========TRAINING==========')
        for label in set(y):
            print(str(label) + ':', y_train.count(label))
        print('==========TESTING==========')
        for label in set(y):
            print(str(label) + ':', y_test.count(label))

        if (SHOW_TEST_IDs):
            print('(Test IDs:', np.unique(test_ids),')')

    train_set = list(zip(X_train, y_train))
    random.shuffle(train_set)
    X_train, y_train = zip(*train_set)

    test_set = list(zip(X_test, y_test))
    random.shuffle(test_set)
    X_test, y_test = zip(*test_set)
    print('')
    return (X_train, X_test, y_train, y_test)

def mlModels(data_train_test):
    X_train = flattenData(data_train_test[0])
    X_test = flattenData(data_train_test[1])
    y_train = np.array(data_train_test[2])
    y_test = np.array(data_train_test[3])

    #X_train, X_test, y_train, y_test = splitById(X, y, labels_id)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)

    model_dc = dc(X_train, X_test, y_train, y_test)
    model_lr = lr(X_train, X_test, y_train, y_test)
    model_svm = svm(X_train, X_test, y_train, y_test)

    return model_dc, model_lr, model_svm

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

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]

    # https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
    model_m = Sequential()
    model_m.add(Conv1D(filter1, kernel, activation='relu', input_shape=(n_timesteps, n_features)))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(filter2, kernel, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(dropout_rate))
    model_m.add(Dense(n_outputs, activation='softmax'))
    #print(model_m.summary())
    if(BINARY): loss = 'binary_crossentropy'
    else: loss = 'categorical_crossentropy'
    model_m.compile(loss=loss,
                    optimizer='adam', metrics=['accuracy'])

    model_m.fit(X_train, y_train, epochs=epochs, verbose=0) # validation_split=TEST_SIZE
    evaluate_nn_summary(model_m, X_test, y_test, disp_labels)
    return model_m

#==================================================================================
#                                   Main
#==================================================================================
DATA = 'data' if LABEL != 'abnormality' else 'data_na'
ID = 'labels_id' if LABEL != 'abnormality' else 'labels_id_na'

with open('..\\classifier_data\\' + DATA + '.pickle', 'rb') as f:
    data = pickle.load(f)
with open('..\\classifier_data\\labels_' + LABEL + '.pickle', 'rb') as f:
    labels = pickle.load(f)
with open('..\\classifier_data\\' + ID + '.pickle', 'rb') as f:
    labels_id = pickle.load(f)

n = REPEAT_AMOUNT if(REPEAT) else 1
for _ in trange(n, ncols=100):
    print()
    if(REPEAT): SEED = random.randint(1, 1000)
    if(SPLIT_BY_ID):
        data_train_test = splitById(data, labels, labels_id)
    else:
        data_train_test = train_test_split(data, labels, test_size=TEST_SIZE, shuffle=True, random_state=SEED)

    mlModels(data_train_test)
    nn(data_train_test)
    if(not REPEAT): print('SEED:', SEED)

if(REPEAT):
    print("\n,\n,\n,\n,\n, \n,\n,\n,\n,\n, \n,\n,\n,\n,\n")
    print('==== Average of ', n, ' times  ========== ')
    print('Base classifier: {:.4f} % accuracy with {:.4f} % deviance'.format(mean(acc_dc) * 100, stdev(acc_dc) * 100))
    print('Logistic Regression: {:.4f} % accuracy with {:.4f} % deviance'.format(mean(acc_lr) * 100, stdev(acc_lr) * 100))
    print('Support Vector Machine: {:.4f} % accuracy with {:.4f} % deviance'.format(mean(acc_svm) * 100, stdev(acc_svm) * 100))
    print('Convolutional Neural Network: {:.4f} % accuracy with {:.4f} % deviance'.format(mean(acc_cnn) * 100, stdev(acc_cnn) * 100))

