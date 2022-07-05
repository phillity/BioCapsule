import os
import numpy as np
from sklearn.svm import SVC

np.random.seed(42)


def get_clf():
    clf = SVC(kernel="linear", probability=True, random_state=42)
    return clf


def train_clf(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def predict_clf(clf, X_test):
    y_prob = clf.predict_proba(X_test)
    return y_prob
