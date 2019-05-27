# Identification Experiment script
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
import sys


def identification(data, data_flip, labels, thread_cnt, data_filename):
    print("Identification")

    # Get k-fold split of dataset (k=5)
    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=1)
    cv.get_n_splits(data, labels)

    ### Perform k-fold cross validation
    y_prob_list = []
    y_pred = np.array([])
    y_true = np.array([])
    for k, (train_index, test_index) in enumerate(cv.split(data, labels)):
        print("     Fold - " + str(k))

        # Get training and testing sets
        train = np.vstack([data[train_index, :], data_flip[train_index, :]])
        train_labels = np.append(labels[train_index], labels[train_index])
        test = data[test_index, :]
        test_labels = labels[test_index]

        # Normalize to z-scores
        mu = np.mean(train, axis=0)
        std = np.std(train, axis=0)
        train = (train - mu) / std
        test = (test - mu) / std

        # Get training classes
        classes = np.unique(train_labels)

        # Training
        svm = SVC(kernel='linear', probability=True)
        svm.fit(train, train_labels)

        # Testing
        prediction = svm.predict(test)
        prob = svm.predict_proba(test)

        for i, label in enumerate(test_labels):
            j = int(label-1)
            y_prob_list.append(prob[i, j])

        y_true = np.append(y_true, test_labels)
        y_pred = np.append(y_pred, prediction)

    print()

    # Overall Results
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(confusion_matrix.shape[0]):
        TP_i = confusion_matrix[i, i]
        FP_i = np.sum(confusion_matrix[i, :]) - TP_i
        FN_i = np.sum(confusion_matrix[:, i]) - TP_i
        TN_i = np.sum(np.sum(confusion_matrix)) - TP_i - FP_i - FN_i

        TP = TP + TP_i
        FP = FP + FP_i
        FN = FN + FN_i
        TN = TN + TN_i

    ACC = (TP + TN) / (TP + TN + FP + FN)
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)

    # Print results
    print(data_filename)
    print("--------------------------------------------------------------------------------------")
    print("Identification Results:")
    print("TP: " + str(TP) + "\n" +
    "FP: " + str(FP) + "\n" +
    "FN: " + str(FN) + "\n" +
    "TN: " + str(TN) + "\n" +
    "ACC: " + str(ACC) + "\n" +
    "FAR: " + str(FAR) + "\n" +
    "FRR: " + str(FRR))
    print(str(min(y_prob_list)))
    print()

#dir = "features//"
dir = "bc//bc_same//"
#dir = "bc//bc_uni//"

database = "caltech"

#mode = "crop"
mode = "align"
#mode = ""

#model = "20180402-114759"
model = "20180408-102900"

if mode == "":
    if dir == "bc//bc_uni//":
        data_path = dir + database + "//" + database + "_" + model + "_bc_uni.txt"
        data_filename = database + "_" + model + "_bc_uni.txt"
    elif dir == "bc//bc_same//":
        data_path = dir + database + "//" + database + "_" + model + "_bc_same.txt"
        data_filename = database + "_" + mode + "_bc_same.txt"
    else:
        data_path = dir + database + "//" + database + "_" + model + ".txt"
        data_filename = database + "_" + model + ".txt"
else:
    if dir == "bc//bc_uni//":
        data_path = dir + database + "//" + database + "_" + mode + "_" + model + "_bc_uni.txt"
        data_filename = database + "_" + mode + "_" + model + "_bc_uni.txt"
    elif dir == "bc//bc_same//":
        data_path = dir + database + "//" + database + "_" + mode + "_" + model + "_bc_same.txt"
        data_filename = database + "_" + mode + "_" + model + "_bc_same.txt"
    else:
        data_path = dir + database + "//" + database + "_" + mode + "_" + model + ".txt"
        data_filename = database + "_" + mode + "_" + model + ".txt"


print(data_path)

print("Loading data")
data = np.loadtxt(data_path)
print("Loading data_flip")
data_flip = np.loadtxt(data_path[:-4] + "_flip.txt")
print("\n\n")


labels =  data[:, -1]
data = data[:, :-1]
data_flip = data_flip[:, :-1]

identification(data,data_flip, labels, 10, data_filename)