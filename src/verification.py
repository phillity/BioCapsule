import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from utils import bc_lfw
from verification_model import get_clf, train_clf, predict_clf


np.random.seed(42)


def verification(method, rs_cnt):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "{}_{}_lfw.txt".format(method, rs_cnt)), "w")

    lfw = bc_lfw(method, rs_cnt)

    acc, fpr, frr, eer, aucs = [np.zeros((10,)) for _ in range(5)]
    y_true, y_prob = [np.zeros((10, 600)) for _ in range(2)]
    for fold in range(10):
        print("BC+LFW Verification -- Fold {} -- RS Count {}".format(fold, rs_cnt))
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
        p_dist = euclidean_distances(
            test_fold[:, 0, :][:, :-1], test_fold[:, 1, :][:, :-1])
        p_dist = p_dist.diagonal()[..., np.newaxis]
        y_true[fold, :] = np.hstack([np.ones((300,)), np.zeros((300,))])

        y_prob[fold, :] = predict_clf(clf, p_dist)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_true[fold, :], np.around(y_prob[fold, :])).ravel()
        acc[fold] = (tp + tn) / (tp + tn + fp + fn)
        fpr[fold] = fp / (fp + tn)
        frr[fold] = fn / (fn + tp)

        fprf, tprf, thresholds = roc_curve(y_true[fold, :], y_prob[fold, :])
        eer[fold] = brentq(lambda x: 1. - x - interp1d(fprf, tprf)(x), 0., 1.)
        aucs[fold] = auc(fprf, tprf)

        out_file.write("Fold {}:\n".format(fold))
        out_file.write("ACC -- {:.6f}\n".format(acc[fold]))
        out_file.write("FPR -- {:.6f}\n".format(fpr[fold]))
        out_file.write("FRR -- {:.6f}\n".format(frr[fold]))
        out_file.write("EER -- {:.6f}\n".format(eer[fold]))
        out_file.write("AUC -- {:.6f}\n".format(aucs[fold]))
        out_file.write("FP -- {}, FN -- {}\n".format(fp, fn))
        out_file.flush()

    out_file.write("Overall:\n")
    out_file.write(
        "ACC -- {:.6f} (+/-{:.6f})\n".format(np.average(acc), np.std(acc)))
    out_file.write(
        "FPR -- {:.6f} (+/-{:.6f})\n".format(np.average(fpr), np.std(fpr)))
    out_file.write(
        "FRR -- {:.6f} (+/-{:.6f})\n".format(np.average(frr), np.std(frr)))
    out_file.write(
        "EER -- {:.6f} (+/-{:.6f})\n".format(np.average(eer), np.std(eer)))
    out_file.write(
        "AUC -- {:.6f} (+/-{:.6f})\n".format(np.average(aucs), np.std(aucs)))
    out_file.close()

    return y_true, y_prob, acc


if __name__ == "__main__":
    for method in ["arcface", "facenet"]:
        for rs_cnt in range(6):
            y_true, y_prob, acc = verification(method, rs_cnt)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_true.npz".format(method, rs_cnt)), y_true)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_prob.npz".format(method, rs_cnt)), y_prob)
            np.savez_compressed(os.path.join(os.path.abspath(
                ""), "results", "{}_{}_lfw_acc.npz".format(method, rs_cnt)), acc)
