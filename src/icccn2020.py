import os
import cv2
import shutil
from queue import Queue
from threading import Thread
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser
from face import ArcFace, extract_dataset
from utils import progress_bar
from biocapsule import BioCapsuleGenerator


np.random.seed(42)


def filter_lfw(features, features_flip):
    y = np.unique(features[:, -1])
    mask = np.ones(features[:, -1].shape, dtype=bool)
    for y_i in y:
        if features[features[:, -1] == y_i].shape[0] < 5:
            idxes = np.where(features[:, -1] == y_i)
            mask[idxes] = False
    features = features[mask]
    features_flip = features_flip[mask]

    y_map = {}
    y = np.unique(features[:, -1])
    for i, y_i in enumerate(y):
        y_map[y_i] = i + 1

    for i in range(features[:, -1].shape[0]):
        features[i, -1] = y_map[features[i, -1]]
        features_flip[i, -1] = y_map[features_flip[i, -1]]

    return features, features_flip


def get_rs_features():
    arcface = ArcFace()

    if os.path.isdir(os.path.join(os.path.abspath(""), "images", "rs_aligned")):
        shutil.rmtree(os.path.join(
            os.path.abspath(""), "images", "rs_aligned"))

    rs_features = np.zeros((6, 512))
    os.mkdir(os.path.join(os.path.abspath(""), "images", "rs_aligned"))
    for s_id, subject in enumerate(os.listdir(os.path.join(os.path.abspath(""), "images", "rs"))[4:]):
        for image in os.listdir(os.path.join(os.path.abspath(""), "images", "rs", subject)):
            img = cv2.imread(os.path.join(
                os.path.abspath(""), "images", "rs", subject, image))
            img_aligned = arcface.preprocess(img)
            feature = arcface.extract(img_aligned, align=False)
            rs_features[s_id] = feature

            if img_aligned.shape != (3, 112, 112):
                img_aligned = cv2.resize(img_aligned, (112, 112))
                img_aligned = np.rollaxis(
                    cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR), 2, 0)

            cv2.imwrite(os.path.join(
                os.path.abspath(""), "images", "rs_aligned", image), cv2.cvtColor(
                np.rollaxis(img_aligned, 0, 3), cv2.COLOR_RGB2BGR))

    return rs_features


def rs_rbac(y, dist):
    if dist == "unbal":
        rs_map = np.random.choice(
            6, y.shape[0], p=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
    else:
        rs_map = np.random.choice(6, y.shape[0])

    return rs_map


def get_bcs(features, features_flip, rs_features, rs_map):
    bcs = np.zeros(
        (rs_features.shape[0], features.shape[0], 513))
    bcs_flip = np.zeros(
        (rs_features.shape[0], features_flip.shape[0], 513))

    bc_gen = BioCapsuleGenerator()
    for i in range(rs_features.shape[0]):
        bcs[i, :, :] = np.hstack([bc_gen.biocapsule_batch(
            features[:, :-1], rs_features[i]), features[:, -1][:, np.newaxis]])
        bcs_flip[i, :, :] = np.hstack([bc_gen.biocapsule_batch(
            features_flip[:, :-1], rs_features[i]), features_flip[:, -1][:, np.newaxis]])

    return bcs, bcs_flip


def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def authenticate(y_i, X_train_i, y_train_i, X_test_i, y_test_i, queue):
    clf = LogisticRegression(
        class_weight="balanced", random_state=42).fit(X_train_i, y_train_i)

    y_pred_i = clf.predict(X_test_i)

    y_pred_i[y_pred_i != y_i] = 0
    y_pred_i[y_pred_i == y_i] = 1

    y_test_i[y_test_i != y_i] = 0
    y_test_i[y_test_i == y_i] = 1

    queue.put(confusion_matrix(y_test_i, y_pred_i).ravel())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, choices=["lfw", "gtdb"],
                        help="dataset to use in experiment")
    parser.add_argument("-m", "--mode", required=True, choices=["under", "bc"],
                        help="feature mode to use in experiment")
    parser.add_argument("-r", "--role_dist", required=False, choices=["bal", "unbal"], default="unbal",
                        help="role distribution to use in experiment")
    parser.add_argument("-t", "--thread_cnt", required=False, type=int, default=1,
                        help="thread count to use in classifier training")
    parser.add_argument("-gpu", "--gpu", required=False, type=int, default=-1,
                        help="gpu to use in feature extraction")
    args = vars(parser.parse_args())

    if args["mode"] == "under":
        fi = open(os.path.join(os.path.abspath(""), "results",
                               "icccn2020_{}_under.txt".format(args["dataset"])), "w")
    else:
        fi = open(os.path.join(os.path.abspath(""), "results",
                               "icccn2020_{}_bc_{}.txt".format(args["dataset"], args["role_dist"])), "w")

    '''
    features, features_flip = extract_dataset(
        args["dataset"], "arcface", args["gpu"])
    '''

    features = np.load(os.path.join(os.path.abspath(""), "data",
                                    "{}_arcface_feat.npz".format(args["dataset"])))["arr_0"]
    features_flip = np.load(os.path.join(os.path.abspath(""), "data",
                                         "{}_arcface_feat_flip.npz".format(args["dataset"])))["arr_0"]

    # remove all subjects with less than 5 images from LFW dataset
    if args["dataset"] == "lfw":
        features, features_flip = filter_lfw(features, features_flip)

    # if biocapsules are used, we can perform authn-authz operation using reference subjects
    if args["mode"] == "bc":
        # get reference subjects for roles
        rs_features = get_rs_features()

        # assign subjects their reference subjects/roles
        rs_map = rs_rbac(np.unique(features[:, -1]), args["role_dist"])
        cnts = np.unique(rs_map, return_counts=True)[1]
        for i, cnt in enumerate(cnts):
            fi.write("Role {} -- {} Subjects\n".format(i + 1, cnt))

        # create all possible biocapsules
        bcs, bcs_flip = get_bcs(features, features_flip, rs_features, rs_map)

    # tn, fp, fn, tp
    conf_mat = np.zeros((4,))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(skf.split(features[:, :-1], features[:, -1])):
        if args["mode"] == "under":
            X_train = np.vstack([features[:, :-1][train_index],
                                 features_flip[:, :-1][train_index]])
            y_train = np.hstack([features[:, -1][train_index],
                                 features_flip[:, -1][train_index]])
            X_test = features[:, :-1][test_index]
            y_test = features[:, -1][test_index]
            labels = np.unique(y_train)

        else:
            X_train = np.zeros(
                (rs_features.shape[0], train_index.shape[0], 512))
            y_train = np.zeros((rs_features.shape[0], train_index.shape[0]))
            X_test = np.zeros(
                (rs_features.shape[0], test_index.shape[0], 512))
            y_test = np.empty((rs_features.shape[0], test_index.shape[0]))
            for i in range(rs_features.shape[0]):
                X_train[i, :, :] = bcs[i, :, :-1][train_index]
                y_train[i, :] = bcs[i, :, -1][train_index]
                X_test[i, :, :] = bcs[i, :, :-1][test_index]
                y_test[i, :] = bcs[i, :, -1][test_index]
            labels = np.unique(y_train[0])

        threads = []
        queue = Queue()
        classes_split = list(split_list(labels.tolist(), args["thread_cnt"]))
        for i, classes in enumerate(classes_split):
            progress_bar("Fold {}/5".format(k + 1), (i + 1) /
                         len(classes_split))
            for y_i in classes:
                if args["mode"] == "under":
                    X_train_i = X_train
                    y_train_i = np.copy(y_train)
                    X_test_i = X_test
                    y_test_i = np.copy(y_test)
                else:
                    X_train_i = X_train[rs_map[int(y_i - 1)]]
                    y_train_i = np.copy(y_train[rs_map[int(y_i - 1)]])
                    X_test_i = X_test[rs_map[int(y_i - 1)]]
                    y_test_i = np.copy(y_test[rs_map[int(y_i - 1)]])   

                threads.append(Thread(target=authenticate, args=(
                    y_i, X_train_i, y_train_i, X_test_i, y_test_i, queue)))
                threads[-1].start()

            _ = [t.join() for t in threads]
            while not queue.empty():
                conf_mat += queue.get()

    # (tn + tp) / (tn + fp + fn + tp)
    acc = (conf_mat[0] + conf_mat[3]) / np.sum(conf_mat)
    # fp / (tn + fp)
    far = conf_mat[1] / (conf_mat[0] + conf_mat[1])
    # fn / (fn + tp)
    frr = conf_mat[2] / (conf_mat[2] + conf_mat[3])

    fi.write("Dataset -- {}\n".format(args["dataset"]))
    fi.write("BC  -- {}\n".format(args["mode"]))
    fi.write("RS  -- {}\n".format(args["role_dist"]))
    fi.write("FP -- {:.6f}\n".format(conf_mat[1]))
    fi.write("FN -- {:.6f}\n".format(conf_mat[2]))
    fi.write("ACC -- {:.6f}\n".format(acc))
    fi.write("FAR -- {:.6f}\n".format(far))
    fi.write("FRR -- {:.6f}\n".format(frr))
    fi.close()
