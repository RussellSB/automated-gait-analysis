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
#                                   Main
#==================================================================================
with open('..\\classifier_data\\data_na.pickle', 'rb') as f:
    data_na = pickle.load(f)
with open('..\\classifier_data\\labels_na.pickle', 'rb') as f:
    labels_na = pickle.load(f)

print(len(data_na), len(labels_na))

X_outliers = []
for i in range(len(labels_na) - 1, 0, -1):
    if(labels_na[i] == 1):
        print(i)
        X_outliers.append(data_na[i][0])
        del labels_na[i]
        del data_na[i]

X = data_na
y = labels_na

X = [x[0] for x in X]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)



