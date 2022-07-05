import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from utils import bc_lfw
from verification_model import get_clf, train_clf, predict_clf


np.random.seed(42)


def verification(method, rs_cnt):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "{}_{}_lfw_leakage.txt".format(method, rs_cnt)), "w")

    lfw = bc_lfw(method, rs_cnt)
    lfw_prev = bc_lfw(method, rs_cnt - 1)

    acc, fpr, frr, eer, aucs = [np.zeros((10,)) for _ in range(5)]
    y_true, y_prob = [np.zeros((10, 1200)) for _ in range(2)]
    for fold in range(10):
        print("BC+LFW Verification Leakage -- Fold {} -- RS Count {}".format(fold, rs_cnt))
        train_fold = lfw["train_{}".format(fold)]
        X_train = train_fold[:, :-1]
        y_train = train_fold[:, -1][..., np.newaxis]

        p_dist = euclidean_distances(X_train, X_train)
        p_dist = p_dist[np.triu_indices_from(p_dist, k=1)][..., np.newaxis]
        y_mat = np.equal(y_train, y_train.T).astype(int)
        y = y_mat[np.triu_indices_from(y_mat, k=1)]

        X_train = np.vstack([p_dist[y == 0][np.random.choice(p_dist[y == 0].shape[0], p_dist[y == 1].shape[0], replace=False)],
                             p_dist[y == 1]])
        y_train = np.hstack(
            [np.zeros((p_dist[y == 1].shape[0],)), np.ones((p_dist[y == 1].shape[0],))])

        clf = get_clf()

        clf = train_clf(clf, X_train, y_train)

        test_fold = lfw["test_{}".format(fold)]
        test_fold_prev = lfw_prev["test_{}".format(fold)]
        X_test = np.vstack(
            [test_fold[:, 0, :][:, :-1], test_fold[:, 1, :][:, :-1]])
        X_test_prev = np.vstack(
            [test_fold_prev[:, 0, :][:, :-1], test_fold_prev[:, 1, :][:, :-1]])
        p_dist = euclidean_distances(X_test, X_test_prev)
        p_dist = p_dist.diagonal()[..., np.newaxis]
        y_true[fold, :] = np.zeros((1200,))

        y_prob[fold, :] = predict_clf(clf, p_dist)[:, 1]

        tn = np.sum(np.around(y_prob[fold, :]) == 0)
        fp = np.sum(np.around(y_prob[fold, :]) == 1)
        acc[fold] = tn / (tn + fp)

        out_file.write("Fold {}:\n".format(fold))
        out_file.write("ACC -- {:.6f}\n".format(acc[fold]))
        out_file.write("TN -- {}, FP -- {}\n".format(tn, fp))
        out_file.flush()

    out_file.write("Overall:\n")
    out_file.write(
        "ACC -- {:.6f} (+/-{:.6f})\n".format(np.average(acc), np.std(acc)))
    out_file.close()

    return y_true, y_prob, acc


if __name__ == "__main__":
    for method in ["arcface", "facenet"]:
        for rs_cnt in range(1, 6):
            y_true, y_prob, acc = verification(method, rs_cnt)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_leakage_true.npz".format(method, rs_cnt)), y_true)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_leakage_prob.npz".format(method, rs_cnt)), y_prob)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_leakage_acc.npz".format(method, rs_cnt)), acc)
