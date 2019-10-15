import os
import csv
import numpy as np
from argparse import ArgumentParser
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from biocapsule import BioCapsuleGenerator


np.random.seed(42)


def get_biocapsules(data, data_flip, rs_data):
    user_y = data[:, -1]
    user_feat = data[:, :-1]
    user_feat_flip = data[:, :-1]

    rs_y = rs_data[:, -1]
    rs_feat = rs_data[:, :-1]

    bc = np.zeros((user_feat.shape[0], 513))
    bc_flip = np.zeros((user_feat_flip.shape[0], 513))
    bc_gen = BioCapsuleGenerator()
    for i, y in enumerate(user_y):
        bc[i] = np.append(bc_gen.biocapsule(user_feat[i, :], rs_feat[0]), y)
        bc_flip[i] = np.append(bc_gen.biocapsule(
            user_feat_flip[i, :], rs_feat[0]), y)
    return bc, bc_flip


def setup():
    path = os.path.dirname(os.path.realpath(__file__))

    people = []
    with open(os.path.abspath(os.path.join(path, "..", "images", "people.txt")), "r") as people_file:
        people_list = list(csv.reader(people_file, delimiter="\t"))
        assert(len(people_list[2:603]) == 601)
        people.append(people_list[2:603])
        assert(len(people_list[604:1159]) == 555)
        people.append(people_list[604:1159])
        assert(len(people_list[1160:1712]) == 552)
        people.append(people_list[1160:1712])
        assert(len(people_list[1713:2273]) == 560)
        people.append(people_list[1713:2273])
        assert(len(people_list[2274:2841]) == 567)
        people.append(people_list[2274:2841])
        assert(len(people_list[2842:3369]) == 527)
        people.append(people_list[2842:3369])
        assert(len(people_list[3370:3967]) == 597)
        people.append(people_list[3370:3967])
        assert(len(people_list[3968:4569]) == 601)
        people.append(people_list[3968:4569])
        assert(len(people_list[4570:5150]) == 580)
        people.append(people_list[4570:5150])
        assert(len(people_list[5151:]) == 609)
        people.append(people_list[5151:])

    pairs = []
    with open(os.path.abspath(os.path.join(path, "..", "images", "pairs.txt")), "r") as pairs_file:
        pairs_list = list(csv.reader(pairs_file, delimiter="\t"))
        for i in range(10):
            idx = i * 300 + 1
            pairs.append(pairs_list[idx: idx + 300])
            assert(len(pairs[i]) == 300)

    return people, pairs


def verification(feat_mode, bc_mode):
    people, pairs = setup()

    path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.abspath(os.path.join(path, "..", feat_mode + "_data", "lfw"))
    data = np.load(dataset_path + "_feat.npz")["arr_0"]
    data_flip = np.load(dataset_path + "_feat_flip.npz")["arr_0"]

    subjects = {}
    for i, subject in enumerate(os.listdir(os.path.abspath(os.path.join(path, "..", "images", "lfw")))):
        subjects[subject] = i + 1

    if bc_mode != "underlying":
        rs_path = os.path.abspath(os.path.join(
            path, "..", feat_mode + "_data", "caltech"))
        rs_data = np.load(rs_path + "_feat.npz")["arr_0"]
        data, data_flip = get_biocapsules(data, data_flip, rs_data)

    X = data[:, :-1]
    X_flip = data_flip[:, :-1]
    y = data[:, -1]

    svms = []
    for k, people_fold in enumerate(people):
        X_train = np.zeros((len(people_fold), 512))
        X_train_flip = np.zeros((len(people_fold), 512))
        y_train = np.zeros((len(people_fold),))
        for p, person in enumerate(people_fold):
            p_y = subjects[person[0]]
            X_train[p] = X[y == p_y][int(person[1]) - 1]
            X_train_flip[p] = X_flip[y == p_y][int(person[1]) - 1]
            y_train[p] = p_y
        X_train = np.vstack([X_train, X_train_flip])
        y_train = np.hstack([y_train, y_train])

        dists = []
        dists_c = []
        for i in range(X_train.shape[0]):
            for j in range(i + 1, X_train.shape[0]):
                dists_c.append(int(y_train[i] == y_train[j]))
                dists.append(np.linalg.norm(X_train[i] - X_train[j]))

        class_weights = compute_class_weight(
            "balanced", np.unique(dists_c), dists_c)
        sample_weights = np.array(dists_c, dtype=float)
        sample_weights[sample_weights == 0] = class_weights[0]
        sample_weights[sample_weights == 1] = class_weights[1]
        svms.append(SVC(gamma="auto", probability=True).fit(np.array(dists).reshape(-1, 1), np.array(dists_c), sample_weight=sample_weights))

    ACC, FAR, FRR, PRE, REC, F1M, EER = [], [], [], [], [], [], []
    for k, pairs_fold in enumerate(pairs):
        dists = []
        dists_c = []
        for p, pair in enumerate(pairs_fold):
            if len(pair) == 3:
                p_y = subjects[pair[0]]
                f_1 = X[y == p_y][int(pair[1]) - 1]
                f_2 = X[y == p_y][int(pair[2]) - 1]
                dists_c.append(1)
                dists.append(np.linalg.norm(f_1 - f_2))
            else:
                p_1 = subjects[pair[0]]
                f_1 = X[y == p_1][int(pair[1]) - 1]
                p_2 = subjects[pair[2]]
                f_2 = X[y == p_2][int(pair[3]) - 1]
                dists_c.append(0)
                dists.append(np.linalg.norm(f_1 - f_2))

        y_prob = svms[k].predict_proba(np.array(dists).reshape(-1, 1))
        y_true = dists_c

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        EER.append(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))

        TN, FP, FN, TP = confusion_matrix(y_true, np.around(y_prob)).ravel()
        ACC.append((TP + TN) / (TP + TN + FP + FN))
        FAR.append(FP / (FP + TN))
        FRR.append(FN / (FN + TP))
        PRE.append(TP / (TP + FP))
        REC.append(TP / (TP + FN))
        F1M.append(2 * TP / (2 * TP + FP + FN))

    out_path = os.path.abspath(os.path.join(
        path, "..", "results", "verification_lfw_" + feat_mode + "_" + bc_mode + ".txt"))
    with open(out_path, "w") as out_file:
        out_file.write(dataset + " " + feat_mode + " " + bc_mode + "\n")
        out_file.write("ACC -- " + str(np.average(ACC) * 100) + "\n")
        out_file.write("FAR -- " + str(np.average(FAR) * 100) + "\n")
        out_file.write("FRR -- " + str(np.average(FRR) * 100) + "\n")
        out_file.write("PRE -- " + str(np.average(PRE) * 100) + "\n")
        out_file.write("REC -- " + str(np.average(REC) * 100) + "\n")
        out_file.write("F1M -- " + str(np.average(F1M) * 100) + "\n")
        out_file.write("EER -- " + str(np.average(EER) * 100) + "\n")
        out_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--feat_mode", required=True, choices=["arcface", "facenet"],
                        help="feat_mode to use in verification")
    parser.add_argument("-b", "--bc_mode", required=True, choices=["underlying", "same"],
                        help="bc_mode to use in verification")
    args = vars(parser.parse_args())

    verification(args["feat_mode"], args["bc_mode"])
