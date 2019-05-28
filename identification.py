# Identification Experiment script
import os
import sys
from argparse import ArgumentParser
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def identification(database, folds, mode):
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
    for k, (train_index, test_index) in enumerate(cv.split(data, labels)):
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
        svm = SVC(kernel='linear')
        svm.fit(train, train_labels)

        # Testing
        y_pred = svm.predict(test)

        # Micro-average of fold results
        mcm = metrics.multilabel_confusion_matrix(test_labels, y_pred)
        tn = np.sum(mcm[:, 0, 0])
        tp = np.sum(mcm[:, 1, 1])
        fn = np.sum(mcm[:, 1, 0])
        fp = np.sum(mcm[:, 0, 1])
        acc.append((tn + tp) / (fp + fn + tn + tp))
        far.append(fp / (fp + tn))
        frr.append(fn / (fn + tp))

    # Average results of all folds
    print('--------------------------------------------------------------------------------------')
    print(database + '  ' + mode)
    print('ACC: ' + str(np.mean(acc) * 100.) + '\n' +
          'FAR: ' + str(np.mean(far) * 100.) + '\n' +
          'FRR: ' + str(np.mean(frr) * 100.))
    print('--------------------------------------------------------------------------------------')
    print()


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-d', '--database', required=True,
                    help='database to use in identification experiment')
parser.add_argument('-f', '--folds', required=False, default=5, type=int, choices=range(1, 100),
                    help='number of folds in cross validation')
args = vars(parser.parse_args())

# Perform identification experiment
identification(args['database'], args['folds'], 'under')
identification(args['database'], args['folds'], 'same')
identification(args['database'], args['folds'], 'uni')
