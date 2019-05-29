import os
import csv
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib



pairs = "pairs.txt"

with open(pairs,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    pairs_list = list(reader)

print("loading thresholds")
thresholds = []
thresholds.append(np.loadtxt("features_trained.txt"))
thresholds.append(np.loadtxt("bc_same_trained.txt"))
thresholds.append(np.loadtxt("bc_uni_trained.txt"))

data = []
y = []

print("loading feature data")
data_i = np.loadtxt("features//lfw//lfw_align_20180408-102900.txt")
y_i = data_i[:,-1]
data_i = data_i[:,:-1]
data.append(data_i)
y.append(y_i)

print("loading bc same data")
data_i = np.loadtxt("bc//bc_same//lfw//lfw_align_20180408-102900_bc_same.txt")
y_i = data_i[:,-1]
data_i = data_i[:,:-1]
data.append(data_i)
y.append(y_i)

print("loading bc uni data")
data_i = np.loadtxt("bc//bc_uni//lfw//lfw_align_20180408-102900_bc_uni.txt")
y_i = data_i[:,-1]
data_i = data_i[:,:-1]
data.append(data_i)
y.append(y_i)

print("loading lfw subjects")
lfw_subjects = json.load(open("lfw_subjects.txt"))


labels = np.array([])
dists = np.array([])

labels_k = []
dists_k = []

acc = []
far = []
frr = []
eer = []
auc = []

matplotlib.rcParams.update({'font.size': 28})

plt.style.use('seaborn-deep')

plt.figure()
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
for t,thresh in enumerate(thresholds):
    if t > 0:
        fpr, tpr, thresholds = metrics.roc_curve(labels,dists,pos_label=0)

        if t == 1:
            plt.plot(fpr, tpr, color='blue', lw=4, label='Underlying System')
        
        if t == 2:
            plt.plot(fpr, tpr, color='green', lw=4, label='Same RS System')

        print("------------------------------")
        print("ACC " + str(np.mean(acc)))
        print("FAR " + str(np.mean(far)))
        print("FRR " + str(np.mean(frr)))
        print("EER " + str(np.mean(eer)))
        print("AUC " + str(np.mean(auc)))
        print("------------------------------")
    
    print("test " + str(t))
    labels = np.array([])
    dists = np.array([])

    labels_k = []
    dists_k = []

    acc = []
    far = []
    frr = []
    eer = []
    auc = []

    k = 0

    for i,ele in enumerate(pairs_list):
        if i == 0:
            continue

        if len(ele) == 1:
            labels = np.append(labels,labels_k)
            dists = np.append(dists,dists_k)

            thresh_k = thresh[k]

            labels_k = np.array(labels_k)
            dists_k = np.array(dists_k)

            pred_k = np.zeros((dists_k.shape[0],))
            pred_k[dists_k < thresh_k] = 1

            TP, FN, FP, TN = metrics.confusion_matrix(labels_k,pred_k,labels=[0,1]).ravel()
            acc.append((TP + TN) / (TP + TN + FP + FN))
            far.append(FP / (FP + TN))
            frr.append(FN / (FN + TP))

            fpr, tpr, thresholds = metrics.roc_curve(labels_k,dists_k,pos_label=0)
            eer_k = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer.append(eer_k)

            dists_k = np.ones(dists_k.shape) - dists_k
            auc.append(metrics.roc_auc_score(labels_k,dists_k))


            k = k + 1

            labels_k = []
            dists_k = []
            continue

        if len(ele) == 3:
            subject = lfw_subjects[ele[0]]

            subject_feats = data[t][y[t] == subject]

            feat_1 = subject_feats[int(ele[1])-1,:]
            feat_2 = subject_feats[int(ele[2])-1,:]

            dist = np.linalg.norm(feat_1 - feat_2)
            dist = 1 - (dist / 100)

            labels_k.append(0)
            dists_k.append(dist)

        if len(ele) == 4:
            subject_1 = lfw_subjects[ele[0]]
            subject_feats = data[t][y[t] == subject_1]
            feat_1 = subject_feats[int(ele[1])-1,:]

            subject_2 = lfw_subjects[ele[2]]
            subject_feats = data[t][y[t] == subject_2]
            feat_2 = subject_feats[int(ele[3])-1,:]

            dist = np.linalg.norm(feat_1 - feat_2)
            dist = 1 - (dist / 100)

            labels_k.append(1)
            dists_k.append(dist)

print("------------------------------")
print("ACC " + str(np.mean(acc)))
print("FAR " + str(np.mean(far)))
print("FRR " + str(np.mean(frr)))
print("EER " + str(np.mean(eer)))
print("AUC " + str(np.mean(auc)))
print("------------------------------")

fpr, tpr, thresholds = metrics.roc_curve(labels,dists,pos_label=0)
plt.plot(fpr, tpr, color='purple', lw=4, label='Unique RS System')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LFW Facial Verification ROC Curve')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.show()