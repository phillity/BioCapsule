# LFW Verification script
import os
import csv
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as pyplot
import matplotlib
from latex_format import latexify


def subject_dict():
    lfw_subjects = {}
    subject_idx = 0
    cur_path = os.path.dirname(__file__)
    database_path = os.path.join(cur_path, 'images', 'lfw')
    for subject in os.listdir(database_path):
        subject_path = os.path.join(database_path, subject)
        subject_idx += 1
        lfw_subjects[subject] = subject_idx
    return lfw_subjects


def eer_threshold_calc(features):
    labels = features[:, -1]
    features = features[:, :-1]
    dists_same = []
    dists_diff = []
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            if j > i:
                if labels[i] == labels[j]:
                    dists_same.append(np.linalg.norm(features[i, :] - features[j, :]))
                else:
                    dists_diff.append(np.linalg.norm(features[i, :] - features[j, :]))
    dists_same = np.asarray(dists_same)
    dists_diff = np.asarray(dists_diff)
    max_dist = np.amax(np.append(dists_same, dists_diff))
    dists_same = 1. - (dists_same / max_dist)
    dists_diff = 1. - (dists_diff / max_dist)

    y_true = np.append(np.ones(dists_same.shape[0]), np.zeros(dists_diff.shape[0]))
    y_prob = np.append(dists_same, dists_diff)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, thresholds)(eer)
    return eer_thresh, max_dist


def test_fold_result(dists_same, dists_diff, threshold):
    dists_same = np.asarray(dists_same)
    dists_diff = np.asarray(dists_diff)

    y_true = np.append(np.ones(dists_same.shape[0]), np.zeros(dists_diff.shape[0]))
    y_prob = np.append(dists_same, dists_diff)
    y_pred = y_prob >= threshold

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc_k = metrics.accuracy_score(y_true, y_pred)
    far_k = fp / (fp + tn)
    frr_k = fn / (fn + tp)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    eer_k = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc_k = metrics.roc_auc_score(y_true, y_prob)
    return y_true, y_prob, y_pred, acc_k, far_k, frr_k, eer_k, auc_k


def verification(mode):
    eer_thresholds, scaling_factors = verification_train(mode)
    all_true, all_prob = verification_test(mode, eer_thresholds, scaling_factors)
    return all_true, all_prob


def verification_train(mode):
    # Load data
    cur_path = os.path.dirname(__file__)
    if mode == 'under':
        data_path = os.path.join(cur_path, 'data', 'lfw.npz')
        data_flip_path = os.path.join(cur_path, 'data', 'lfw_flip.npz')
    elif mode == 'same':
        data_path = os.path.join(cur_path, 'data', 'lfw_bc_same.npz')
        data_flip_path = os.path.join(cur_path, 'data', 'lfw_flip_bc_same.npz')
    else:
        data_path = os.path.join(cur_path, 'data', 'lfw_bc_uni.npz')
        data_flip_path = os.path.join(cur_path, 'data', 'lfw_flip_bc_uni.npz')
    data = np.load(data_path)['arr_0']
    data_flip = np.load(data_flip_path)['arr_0']
    labels = data[:, -1]

    with open('people.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        people_list = list(reader)

    lfw_subjects = np.load('lfw_subjects.npz')['arr_0'].item()
    feats = np.empty((0, 513))
    eer_thresholds = []
    scaling_factors = []
    for i, ele in enumerate(people_list):
        if len(ele) == 1:
            if feats.shape[0] > 0:
                eer_threshold, scaling_factor = eer_threshold_calc(feats)
                eer_thresholds.append(eer_threshold)
                scaling_factors.append(scaling_factor)

            feats = np.empty((0, 513))
            continue

        label = lfw_subjects[ele[0]]
        idx = np.where(labels == label)[0]
        feat = data[idx, :]
        feat_flip = data_flip[idx, :]
        feats = np.vstack([feats, feat])
        feats = np.vstack([feats, feat_flip])

    return eer_thresholds, scaling_factors


def verification_test(mode, eer_thresholds, scaling_factors):
    all_true = np.array([])
    all_prob = np.array([])

    # Load data
    cur_path = os.path.dirname(__file__)
    if mode == 'under':
        data_path = os.path.join(cur_path, 'data', 'lfw.npz')
    elif mode == 'same':
        data_path = os.path.join(cur_path, 'data', 'lfw_bc_same.npz')
    else:
        data_path = os.path.join(cur_path, 'data', 'lfw_bc_uni.npz')
    data = np.load(data_path)['arr_0']
    labels = data[:, -1]

    with open('pairs.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        pairs_list = list(reader)

    lfw_subjects = np.load('lfw_subjects.npz')['arr_0'].item()
    dists_same = []
    dists_diff = []
    acc = []
    far = []
    frr = []
    eer = []
    auc = []
    k = 0
    for i, ele in enumerate(pairs_list):
        if len(ele) < 3:
            if len(dists_same) > 0:
                y_true, y_prob, y_pred, acc_k, far_k, frr_k, eer_k, auc_k = test_fold_result(dists_same, dists_diff, threshold)
                acc.append(acc_k)
                far.append(far_k)
                frr.append(frr_k)
                eer.append(eer_k)
                auc.append(auc_k)
                all_true = np.append(all_true, y_true)
                all_prob = np.append(all_prob, y_prob)

                k += 1

            dists_same = []
            dists_diff = []
            continue

        factor = scaling_factors[k]
        threshold = eer_thresholds[k]

        if len(ele) == 3:
            label = lfw_subjects[ele[0]]
            idx = np.where(labels == label)[0]
            feats = data[idx, :]
            feat_1 = feats[int(ele[1]) - 1, :-1]
            feat_2 = feats[int(ele[2]) - 1, :-1]

            dist = max(1. - (np.linalg.norm(feat_1 - feat_2) / factor), 0.)
            dists_same.append(dist)

        else:
            label = lfw_subjects[ele[0]]
            idx = np.where(labels == label)[0]
            feats = data[idx, :]
            feat_1 = feats[int(ele[1]) - 1, :-1]

            label = lfw_subjects[ele[2]]
            idx = np.where(labels == label)[0]
            feats = data[idx, :]
            feat_2 = feats[int(ele[3]) - 1, :-1]

            dist = max(1. - (np.linalg.norm(feat_1 - feat_2) / factor), 0.)
            dists_diff.append(dist)

    # Average results of all folds
    print('--------------------------------------------------------------------------------------')
    print('lfw  ' + mode)
    print('ACC: ' + str(np.mean(acc) * 100.) + '\n' +
          'FAR: ' + str(np.mean(far) * 100.) + '\n' +
          'FRR: ' + str(np.mean(frr) * 100.) + '\n' +
          'EER: ' + str(np.mean(eer) * 100.) + '\n' +
          'AUC: ' + str(np.mean(auc)))
    print('--------------------------------------------------------------------------------------')
    print()

    return all_true, all_prob


# Perform verification experiment
under_true, under_prob = verification('under')
same_true, same_prob = verification('same')
uni_true, uni_prob = verification('uni')

cur_path = os.path.dirname(__file__)
np.savez_compressed(os.path.join(cur_path, 'data', 'under_true.npz'), under_true)
np.savez_compressed(os.path.join(cur_path, 'data', 'under_prob.npz'), under_prob)
np.savez_compressed(os.path.join(cur_path, 'data', 'same_true.npz'), same_true)
np.savez_compressed(os.path.join(cur_path, 'data', 'same_prob.npz'), same_prob)
np.savez_compressed(os.path.join(cur_path, 'data', 'uni_true.npz'), uni_true)
np.savez_compressed(os.path.join(cur_path, 'data', 'uni_prob.npz'), uni_prob)

'''
# Generate ROC
latexify()

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
fpr, tpr, thresholds = metrics.roc_curve(under_true, under_prob, pos_label=1)
pyplot.plot(fpr, tpr, color='blue', lw=4, label='Underlying System')
fpr, tpr, thresholds = metrics.roc_curve(same_true, same_prob, pos_label=1)
pyplot.plot(fpr, tpr, color='green', lw=4, label='Same RS System')
fpr, tpr, thresholds = metrics.roc_curve(uni_true, uni_prob, pos_label=1)
pyplot.plot(fpr, tpr, color='purple', lw=4, label='Unique RS System')

pyplot.title('LFW Facial Verification ROC Curve')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend(loc='lower right')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.tight_layout()
# pyplot.show()
pyplot.savefig('lfw_roc.pdf')
'''
