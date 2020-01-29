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

        print("Fold {}:".format(fold))
        print("ACC -- {:.6f}".format(acc[fold]))
        print("FPR -- {:.6f}".format(fpr[fold]))
        print("FRR -- {:.6f}".format(frr[fold]))
        print("EER -- {:.6f}".format(eer[fold]))
        print("AUC -- {:.6f}".format(aucs[fold]))
        print("FP -- {}, FN -- {}".format(fp, fn))

    def p(metric): return (np.average(metric), np.std(metric))
    print("Overall:")
    print("ACC -- {:.6f} (+/-{:.6f})".format(p(acc)))
    print("FPR -- {:.6f} (+/-{:.6f})".format(p(fpr)))
    print("FRR -- {:.6f} (+/-{:.6f})".format(p(frr)))
    print("EER -- {:.6f} (+/-{:.6f})".format(p(err)))
    print("AUC -- {:.6f} (+/-{:.6f})".format(p(aucs)))

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
