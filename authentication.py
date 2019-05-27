# Authentication Experiment script
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

def split_list(l,n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

### Authentication training thread function
def authentication_train(c,train,train_labels,que):
    print("          Training - Class " + str(c))
    binary_labels = np.zeros(train_labels.shape)
    binary_labels[train_labels != c] = 1

    svm = SVC(kernel='linear', probability=True)
    svm.fit(train,binary_labels)

    que.put((c,svm))

### Authentication testing thread function
def authentication_test(c,svm,test,test_labels,que):
    print("          Testing - Class " + str(c))
    
    binary_labels = np.zeros(test_labels.shape)
    binary_labels[test_labels != c] = 1

    # Use binary class svm to predict accept/reject test set
    prediction_prob = svm.predict_proba(test)

    result = []
    result.append(prediction_prob[:,0])
    result.append(binary_labels)
    result.append(c)

    que.put(result)

### AUTHENTICATION
def authentication(data,data_flip,labels,thread_cnt,data_filename):
    print("Authentication")

    # Get k-fold split of dataset (k=5)
    cv = StratifiedKFold(n_splits=2,shuffle=False,random_state=0)
    cv.get_n_splits(data,labels)

    ### Perform k-fold cross validation
    y_prob = np.array([])
    y_pred = np.array([])
    y_true = np.array([])
    for k,(train_index,test_index) in enumerate(cv.split(data,labels)):
        print("     Fold - " + str(k))

        # Get training and testing sets
        train = np.vstack([data[train_index,:],data_flip[train_index,:]])
        train_labels = np.append(labels[train_index],labels[train_index])
        test = data[test_index,:]
        test_labels = labels[test_index]

        # Normalize to z-scores
        mu = np.mean(train,axis=0)
        std = np.std(train,axis=0)
        train = (train - mu) / std
        test = (test - mu) / std

        # Get training classes
        classes = np.unique(train_labels)
        classes_split = list(split_list(classes.tolist(),thread_cnt))

        ### TRAINING
        # Binary SVM for each class
        class_svms = []
        c_idxes = []
        threads = []
        que = Queue()

        # Thread to train each class binary SVM
        for li in classes_split:
            for i,c in enumerate(li):
                threads.append(Thread(target=authentication_train,args=(c,train,train_labels,que)))
                threads[-1].start()
            
            # Collect training thread results
            _ = [ t.join() for t in threads ]
            while not que.empty():
                (c_idx,svm) = que.get()
                c_idxes.append(c_idx)
                class_svms.append(svm)

        ### TESTING
        threads = []
        que = Queue()
        for li in classes_split:
            for i,c in enumerate(li):
                c_idx = c_idxes.index(c)
                threads.append(Thread(target=authentication_test,args=(c,class_svms[c_idx],test,test_labels,que)))
                threads[-1].start()

            # Collect testing thread results
            _ = [ t.join() for t in threads ]
            while not que.empty():
                result = que.get()

                c = int(result[2])
                c_prob = result[0]
                c_true = result[1]
                c_pred = np.zeros(c_prob.shape[0])
                c_pred[c_prob<0.5] = 1

                y_prob = np.append(y_prob,c_prob)
                y_true = np.append(y_true,c_true)
                y_pred = np.append(y_pred,c_pred)
    
    print()

    ### OVERALL RESULTS    
    TP, FN, FP, TN = metrics.confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)

    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_prob,pos_label=0)
    EER = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    EER_thresh = interp1d(fpr, thresholds)(EER)
    y_prob = np.ones(y_prob.shape) - y_prob
    AUC = metrics.roc_auc_score(y_true,y_prob)
    
    # Print results
    print(data_filename)
    print("--------------------------------------------------------------------------------------")
    print("Authentication Results:")
    print("TP: " + str(TP) + "\n" +
    "FP: " + str(FP) + "\n" +
    "FN: " + str(FN) + "\n" +
    "TN: " + str(TN) + "\n" +
    "ACC: " + str(ACC) + "\n" +
    "FAR: " + str(FAR) + "\n" +
    "FRR: " + str(FRR) + "\n" +
    "AUC: " + str(AUC) + "\n" +
    "EER: " + str(EER) + "\n" +
    "EER_thresh " + str(EER_thresh))
    print()


#dir = "features//"
#dir = "bc//bc_same//"
dir = "bc//bc_uni//"

database = "orl"

#mode = "crop"
#mode = "align"
mode = ""

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


labels =  data[:,-1]
data = data[:,:-1]
data_flip = data_flip[:,:-1]

authentication(data,data_flip,labels,10,data_filename)

