# Authentication Experiment script
import os
import sys
from argparse import ArgumentParser
from threading import Thread
from queue import Queue
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# Authentication training thread function
def authentication_train(c, train, train_labels, que):
    binary_labels = np.ones(train_labels.shape)
    binary_labels[train_labels != c] = 0

    svm = SVC(kernel='linear', probability=True)
    svm.fit(train, binary_labels)

    que.put((c, svm))


# Authentication testing thread function
def authentication_test(c, svm, test, test_labels, que):
    binary_labels = np.ones(test_labels.shape)
    binary_labels[test_labels != c] = 0

    prediction_prob = svm.predict_proba(test)

    result = []
    result.append(prediction_prob[:, 1])
    result.append(binary_labels)
    result.append(c)
    que.put(result)


def authentication(database, folds, mode, thread_cnt):
    # Load data
    cur_path = os.path.dirname(__file__)
    if mode == 'under':
        data_path = os.path.join(cur_path, 'data', database + '.npz')
        data_flip_path = os.path.join(cur_path, 'data', database + '_flip.npz')
    elif mode == 'same':
        data_path = os.path.join(cur_path, 'data', database + '_bc_same.npz')
        data_flip_path = os.path.join(cur_path, 'data', database + '_flip_bc_same.npz')
    else:
        data_path = os.path.join(cur_path, 'data', database + '_bc_uni.npz')
        data_flip_path = os.path.join(cur_path, 'data', database + '_flip_bc_uni.npz')
    data = np.load(data_path)['arr_0']
    data_flip = np.load(data_flip_path)['arr_0']

    labels = data[:, -1]
    data = data[:, :-1]
    data_flip = data_flip[:, :-1]

    # Get k-fold split of dataset
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    cv.get_n_splits(data, labels)

    # Perform k-fold cross validation
    acc = []
    far = []
    frr = []
    eer = []
    auc = []
    eer_thresh = []
    for k, (train_index, test_index) in enumerate(cv.split(data, labels)):
        y_prob = np.array([])
        y_pred = np.array([])
        y_true = np.array([])

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
        classes_split = list(split_list(classes.tolist(), thread_cnt))

        # Training
        # Binary SVM for each class
        class_svms = []
        c_idxes = []
        threads = []
        que = Queue()

        # Thread to train each class binary SVM
        for li in classes_split:
            for i, c in enumerate(li):
                threads.append(Thread(target=authentication_train, args=(c, train, train_labels, que)))
                threads[-1].start()

            # Collect training thread results
            _ = [t.join() for t in threads]
            while not que.empty():
                (c_idx, svm) = que.get()
                c_idxes.append(c_idx)
                class_svms.append(svm)

        # Testing
        threads = []
        que = Queue()
        for li in classes_split:
            for i, c in enumerate(li):
                c_idx = c_idxes.index(c)
                threads.append(Thread(target=authentication_test, args=(c, class_svms[c_idx], test, test_labels, que)))
                threads[-1].start()

            # Collect testing thread results
            _ = [t.join() for t in threads]
            while not que.empty():
                result = que.get()

                c = int(result[2])
                c_prob = result[0]
                c_true = result[1]
                c_pred = np.zeros(c_prob.shape[0])
                c_pred[c_prob >= 0.5] = 1

                y_prob = np.append(y_prob, c_prob)
                y_pred = np.append(y_pred, c_pred)
                y_true = np.append(y_true, c_true)

        # Micro-average of fold results
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        acc.append(metrics.accuracy_score(y_true, y_pred))
        far.append(fp / (fp + tn))
        frr.append(fn / (fn + tp))

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
        eer.append(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))
        auc.append(metrics.roc_auc_score(y_true, y_prob))

    # Average results of all folds
    print('--------------------------------------------------------------------------------------')
    print(database + '  ' + mode)
    print('ACC: ' + str(np.mean(acc) * 100.) + '\n' +
          'FAR: ' + str(np.mean(far) * 100.) + '\n' +
          'FRR: ' + str(np.mean(frr) * 100.) + '\n' +
          'EER: ' + str(np.mean(eer) * 100.) + '\n' +
          'AUC: ' + str(np.mean(auc)))
    print('--------------------------------------------------------------------------------------')
    print()

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to use in authentication experiment')
parser.add_argument('-f', '--folds', required=False, default=5, type=int, choices=range(1, 100),
                    help='number of folds in cross validation')
parser.add_argument('-t', '--threads', required=False, default=10, type=int, choices=range(1, 1000),
                    help='number of worker threads')
args = vars(parser.parse_args())

# Perform authentication experiment
authentication(args['database'], args['folds'], 'under', args['threads'])
authentication(args['database'], args['folds'], 'same', args['threads'])
authentication(args['database'], args['folds'], 'uni', args['threads'])
