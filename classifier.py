#==================================================================================
#                               CLASSIFIER
#----------------------------------------------------------------------------------
#               Input: Pre-processed data, Output: Classification
#               Classifies according to testing and training data
#==================================================================================
#                                   Imports
#==================================================================================
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D


#==================================================================================
#                                   Methods
#==================================================================================
# Define and Evaluate a Logistic Regression Model
def lr(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    evaluate_lr_summary(classifier, X_test, y_test)

    # Final accuracy on test set
    score = classifier.score(X_test, y_test)
    print('LR-Accuracy: {:.4f}'.format(score))
    return classifier

# Evaluates the logistic regression model w.r.t confusion matrix and ground-truth metrics
def evaluate_lr_summary(classifier, X_test, y_test):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    predicted = classifier.predict(X_test)
    print("Classification report for LR classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test, include_values=False, cmap=plt.cm.Blues)
    disp.figure_.suptitle("")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)
    plt.show()
    plt.close()

# Evaluates the dense neural network model w.r.t confusion matrix and ground-truth metrics
def evaluate_nn_summary(model, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    import numpy as np
    predictions = model.predict(X_test)
    predictedClasses = np.where(predictions > 0.5, 1, 0)
    cm = metrics.confusion_matrix(y_test, predictedClasses)
    print('cm-nn:', cm)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.colorbar()
    plt.show()
    print('Classification Report NN:')
    print(metrics.classification_report(y_test, predictedClasses))

#==================================================================================
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data_na.pickle', 'rb') as f:
    data = pickle.load(f)
with open('..\\classifier_data\\labels_na.pickle', 'rb') as f:
    labels = pickle.load(f)

X = np.array([np.array(x).transpose() for x in data])  # (samples, time-steps, features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

###
verbose = 1
epochs = 20
batch_size = 64
n_outputs = 2
n_timesteps, n_features = X.shape[1], X.shape[2]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit network
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print(accuracy)
evaluate_nn_summary(model, X_test, y_test)