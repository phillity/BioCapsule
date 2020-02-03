import os
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.class_weight import compute_sample_weight
from argparse import ArgumentParser
from utils import bc_lfw


np.random.seed(42)


def verification(method, rs_cnt):
    out_file = open(os.path.join(os.path.abspath(""), "results",
                                 "{}_{}_lfw.txt".format(method, rs_cnt)), "w")

    lfw = bc_lfw(method, rs_cnt)

    acc, fpr, frr, eer, aucs = [np.zeros((10,)) for _ in range(5)]
    y_true, y_prob = [np.zeros((10, 600)) for _ in range(2)]
    for fold in range(10):
        train_fold = lfw["train_{}".format(fold)]
        X_train = train_fold[:, :-1]
        y_train = train_fold[:, -1][..., np.newaxis]

        p_dist = euclidean_distances(X_train, X_train)
        p_dist = p_dist[np.triu_indices_from(p_dist, k=1)][..., np.newaxis]

        y_mat = np.equal(y_train, y_train.T).astype(int)
        y = y_mat[np.triu_indices_from(y_mat, k=1)]
        y_weights = compute_sample_weight("balanced", y)

        svm = SVC(kernel="linear", probability=True, random_state=42).fit(
            p_dist, y, y_weights)

        test_fold = lfw["test_{}".format(fold)]
        p_dist = euclidean_distances(
            test_fold[:, 0, :][:, :-1], test_fold[:, 1, :][:, :-1])
        p_dist = p_dist.diagonal()[..., np.newaxis]

        y_true[fold, :] = np.hstack([np.ones((300,)), np.zeros((300,))])
        y_prob[fold, :] = svm.predict_proba(p_dist)[:, 1]

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
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", required=True, choices=["arcface", "facenet"],
                        help="method to use in feature extraction")
    parser.add_argument("-rs", "--rs_cnt", default=0, type=int, choices=range(0, 11),
                        help="number of rs in biocapsule generation")
    args = vars(parser.parse_args())

    y_true, y_prob, acc = verification(args["method"], args["rs_cnt"])
    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "results", "{}_{}_lfw_true.npz".format(args["method"], args["rs_cnt"])), y_true)
    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "results", "{}_{}_lfw_prob.npz".format(args["method"], args["rs_cnt"])), y_prob)
    np.savez_compressed(os.path.join(os.path.abspath(
        ""), "results", "{}_{}_lfw_acc.npz".format(args["method"], args["rs_cnt"])), acc)
