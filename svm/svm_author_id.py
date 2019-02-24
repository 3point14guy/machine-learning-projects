#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# # linear kernel will have a faster training time
# clf = SVC(kernel="linear")
# # rbf will have lower accuracy at low C values
clf = SVC(kernel="rbf", C=10000.0)
# # the following two lines cuase just 10% of the training data to be used
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t0, 3), "s"

count = 0
for num in pred:
    if num == 1:
        count += 1

print count


acc = accuracy_score(pred, labels_test)
print acc
#########################################################
