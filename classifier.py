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
from sklearn import svm

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

#==================================================================================
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data_na.pickle', 'rb') as f:
    data_na = pickle.load(f)
with open('..\\classifier_data\\labels_na.pickle', 'rb') as f:
    labels = pickle.load(f)

X = [x[0] for x in data_na]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
lr = lr(X_train, X_test, y_train, y_test)
evaluate_lr_summary(lr, X_test, y_test)



